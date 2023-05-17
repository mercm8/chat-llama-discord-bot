from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import random
import logging
import json
import re
import glob
import os
import warnings
import discord
from discord.ext import commands
from discord import app_commands
import torch
import io
import base64
import yaml
from PIL import Image, PngImagePlugin
import requests
import sqlite3

### Edit config.py to include your Discord token.
import config
TOKEN = config.discord['TOKEN'] 
A1111 = "http://127.0.0.1:7860"

logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s (Line: %(lineno)d in %(funcName)s, %(filename)s )',
                    datefmt="%a, %d %b %Y %H:%M:%S +0000", 
                    level=logging.DEBUG)

# Intercept custom bot arguments
import sys
bot_arg_list = ["--limit-history", "--token"]
bot_argv = []
for arg in bot_arg_list:
    try:
        index = sys.argv.index(arg)
    except:
        index = None
    
    if index is not None:
        bot_argv.append(sys.argv.pop(index))
        bot_argv.append(sys.argv.pop(index))

import argparse
parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument("--token", type=str, help="Discord bot token to use their API.")
parser.add_argument("--limit-history", type=int, help="When the history gets too large, performance issues can occur. Limit the history to improve performance.")
bot_args = parser.parse_args(bot_argv)

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="You have modified the pretrained model configuration to control generation")

import modules.extensions as extensions_module
from modules.chat import chatbot_wrapper, clear_chat_log, load_character 
from modules import shared
shared.args.chat = True
from modules.LoRA import add_lora_to_model
from modules.models import load_model

prompt = "This is a conversation with your Assistant. The Assistant is very helpful and is eager to chat with you and answer your questions."
your_name = "You"
llamas_name = "Assistant"

reply_embed_json = {
    "title": "Reply #X",
    "color": 39129,
    "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
    "url": "https://github.com/xNul/chat-llama-discord-bot",
    "footer": {
        "text": "Contribute to ChatLLaMA on GitHub!",
    },
    "fields": [
        {
            "name": your_name,
            "value": ""
        },
        {
            "name": llamas_name,
            "value": ":arrows_counterclockwise:"
        }
    ]
}
reply_embed = discord.Embed().from_dict(reply_embed_json)

reset_embed_json = {
    "title": "Conversation has been reset",
    "description": "Replies: 0\nYour name: " + your_name + "\nLLaMA's name: " + llamas_name + "\nPrompt: " + prompt,
    "color": 39129,
    "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
    "url": "https://github.com/xNul/chat-llama-discord-bot",
    "footer": {
        "text": "Contribute to ChatLLaMA on GitHub!"
    }
}

reset_embed = discord.Embed().from_dict(reset_embed_json)

status_embed_json = {
    "title": "Status",
    "description": "You don't have a job queued.",
    "color": 39129,
    "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
    "url": "https://github.com/xNul/chat-llama-discord-bot",
    "footer": {
        "text": "Contribute to ChatLLaMA on GitHub!"
    }
}
status_embed = discord.Embed().from_dict(status_embed_json)

greeting_embed_json = {
    "title": "",
    "description": "",
    "thumbnail": ""
}
greeting_embed = discord.Embed().from_dict(greeting_embed_json)

info_embed_json = {
    "title": "How to use",
    "description": """
      **/character** - Change character
      **/main** - Set main channel for bot so it can reply without being called by name
      **/pic** - Ask the bot to take a picture. You can also directly ask it to *take a picture* or *take a selfie* in clear text.
      """
}
info_embed = discord.Embed().from_dict(info_embed_json)



# Load text-generation-webui
# Define functions
def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub("-np$", "", item.name) for item in list(Path(f"{shared.args.model_dir}/").glob("*")) if item.name.endswith("-np")], key=str.lower)
    else:
        return sorted([re.sub(".pth$", "", item.name) for item in list(Path(f"{shared.args.model_dir}/").glob("*")) if not item.name.endswith((".txt", "-np", ".pt", ".json", ".yaml"))], key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x: x.parts[1], Path("extensions").glob("*/script.py"))), key=str.lower)

def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings

def list_model_elements():
    elements = ["cpu_memory", "auto_devices", "disk", "cpu", "bf16", "load_in_8bit", "wbits", "groupsize", "model_type", "pre_layer"]
    for i in range(torch.cuda.device_count()):
        elements.append(f"gpu_memory_{i}")
    return elements

# Update the command-line arguments based on the interface values
def update_model_parameters(state, initial=False):
    elements = list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith("gpu_memory"):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ["wbits", "groupsize", "model_type"] and value == "None":
            value = vars(shared.args_defaults)[element]
        elif element in ["cpu_memory"] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ["wbits", "groupsize", "pre_layer"]:
            value = int(value)
        elif element == "cpu_memory" and value is not None:
            value = f"{value}MiB"

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)["gpu_memory"] != vars(shared.args_defaults)["gpu_memory"]):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None

# Loading custom settings
settings_file = None
if shared.args.settings is not None and Path(shared.args.settings).exists():
    settings_file = Path(shared.args.settings)
elif Path("settings.json").exists():
    settings_file = Path("settings.json")
if settings_file is not None:
    print(f"Loading settings from {settings_file}...")
    new_settings = json.loads(open(settings_file, "r").read())
    for item in new_settings:
        shared.settings[item] = new_settings[item]

# Default extensions
extensions_module.available_extensions = get_available_extensions()
if shared.is_chat():
    for extension in shared.settings["chat_default_extensions"]:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
else:
    for extension in shared.settings["default_extensions"]:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

available_models = get_available_models()

# Model defined through --model
if shared.args.model is not None:
    shared.model_name = shared.args.model

# Only one model is available
elif len(available_models) == 1:
    shared.model_name = available_models[0]

# Select the model from a command-line menu
elif shared.model_name == "None" or shared.args.model_menu:
    if len(available_models) == 0:
        print("No models are available! Please download at least one.")
        sys.exit(0)
    else:
        print("The following models are available:\n")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input()) - 1
        print()
    shared.model_name = available_models[i]

# If any model has been selected, load it
if shared.model_name != "None":

    model_settings = get_model_specific_settings(shared.model_name)
    shared.settings.update(model_settings)  # hijacking the interface defaults
    update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

    # Load the model
    shared.model, shared.tokenizer = load_model(shared.model_name)
    if shared.args.lora:
        add_lora_to_model([shared.args.lora])

# Loading the bot
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=".", intents=intents)

queues = []
blocking = False
reply_count = 0

def ceil_timedelta(td):
    return td + timedelta(minutes=1) - timedelta(seconds=td.seconds % 60)
    #return td + (datetime.min - td) % timedelta(minutes=1)
    #return str(td + timedelta(minutes=1) - timedelta(seconds=td.seconds % 60))

async def change_profile(ctx, character):
    """ Changes username and avatar of bot. """
    """ Will be rate limited by discord api if used too often. Needs a cooldown. 10 minute value is arbitrary. """
    #name1, name2, picture, greeting, context, end_of_turn, chat_html_wrapper = load_character(character, '', '', '', '')
    if hasattr(ctx.bot, "last_change"):
        if datetime.now() >= ctx.bot.last_change + timedelta(minutes=10):
            remaining_cooldown = ctx.bot.last_change + timedelta(minutes=10) - datetime.now() 
            await ctx.channel.send(f'Please wait {ceil_timedelta(remaining_cooldown)} before changing character again')
    else:
        try:
            if (ctx.bot.behavior.change_username_with_character and ctx.bot.user.display_name != character):
                await client.user.edit(username=character)
            if (ctx.bot.behavior.change_avatar_with_character):
                folder = 'characters'
                picture_path = os.path.join(folder, f'{character}.png')
                if os.path.exists(picture_path):
                    with open(picture_path, 'rb') as f:
                        picture = f.read()
                    await client.user.edit(avatar=picture)
            new_char = load_character(character, '', '', '', 'cai-chat')
            greeting = new_char[3]
            ctx.bot.llm_context = new_char[4]
            file = discord.File(picture_path, filename=f'{character}.png')
            greeting_embed.title=character
            greeting_embed.description=greeting
            #greeting_embed.set_thumbnail(url=f"attachment://{character}.png")
            greeting_embed.set_image(url=f"attachment://{character}.png")
            await ctx.channel.send(file=file, embed=greeting_embed)
            ctx.bot.last_change = datetime.now()
        except discord.HTTPException as e:
            """ This exception can happen when you restart the bot and change character too fast without last_change being set """
            logging.warning(e)
        except Exception as e:
            logging.warning(e)

    if ctx.bot.behavior.read_chatlog:
        """  Allow bot to read recent chatlog. Might want to do this somewhere else. 
        Context is being fed in load_character which is external. 
        Maybe insert it in shared.history from here? 
        Need to find out how that works. """
        pass


async def send_long_message(channel, message_text):
    """ Splits a longer message into parts, making sure code blocks are maintained """
    codeblock_index = message_text.find("```")
    if codeblock_index >= 0:
        closing_codeblock_index = message_text.find("```", codeblock_index+3)
    
    if len(message_text) <= 2000 or codeblock_index == -1 or closing_codeblock_index == -1:
        await channel.send(message_text)
    else:
        chunk_text = message_text[0:closing_codeblock_index+3]
        await channel.send(chunk_text)
        await send_long_message(channel, message_text[closing_codeblock_index+3:])

async def llm_gen(message, queues):
    global blocking
    global reply_count

    if len(queues) > 0:
        blocking = True
        reply_count += 1
        user_input = queues.pop(0)
        mention = list(user_input.keys())[0]
        user_input = user_input[mention]
        last_resp = ""
        for resp in chatbot_wrapper(**user_input):
            resp_clean = resp[len(resp)-1][1]
            last_resp = resp_clean

        logging.info("reply sent: \"" + mention + ": {'text': '" + user_input["text"] + "', 'response': '" + last_resp + "'}\"")
        await send_long_message(message.channel, last_resp)
        
        if bot_args.limit_history is not None and len(shared.history['visible']) > bot_args.limit_history:
            shared.history['visible'].pop(0)
            shared.history['internal'].pop(0)
        
        await llm_gen(message, queues)
    else:
        blocking = False

@client.event
async def on_ready():
    if not hasattr(client, 'llm_context'):
        """ Loads character profile based on Bot's display name """
        client.llm_context = load_character(client.user.display_name, '', '', '')[4]
    if not hasattr(client, 'behavior'):
        client.behavior = Behavior()
    logging.info("bot ready")
    await client.tree.sync()

def a1111_online():
    try:
        r = requests.get(f'{A1111}/')
        status = r.raise_for_status()
        #logging.info(status)
        return True
    except Exception as exc:
        logging.warning(exc)
        return False
    
async def create_image_prompt(llm_prompt):
    user_input = LLMUserInputs().settings
    user_input["text"] = llm_prompt
    user_input["state"]["name1"] = ""
    user_input["state"]["name2"] = client.user.display_name
    user_input["state"]["context"] = client.llm_context
    
    for resp in chatbot_wrapper(**user_input):
        resp_clean = resp[len(resp)-1][1]
        last_resp = resp_clean
    return last_resp


@client.event
async def on_message(message):
    text = message.clean_content
    ctx = await client.get_context(message)

    if client.behavior.main_channels == None and client.user.mentioned_in(message):
        """ User has not set a main channel for the bot, but is speaking to it. 
        Likely first time use. Setting current channel as main channel for bot which will
        also instruct user on how to change main channel in the embed notification """
        main(ctx)
    image_triggers = ['take a picture', 'take another picture','generate an image','take a selfie','take another selfie'] 
    # Might want to move these triggers into the yaml file to let users localize/customize
    if (any(word in text.lower() for word in image_triggers) or \
        (random.random() < client.behavior.reply_with_image)) \
        and a1111_online() and client.behavior.bot_should_reply(message):

        """ Triggering the LLM here so it's aware of the picture it's sending to the user,
        or else it gets 'confused' when the user responds to the image. """

        if 'selfie' in text:
            llm_prompt = f"""[SYSTEM] You have been tasked with taking a selfie: "{text}".
            Include your appearance, your current state of clothing, your surroundings 
            and what you are doing right now.
            """
        else: 
            llm_prompt = f"""[SYSTEM] You have been tasked with generating an image: "{text}"."""

        llm_prompt += """Describe the image in vivid detail as if you were describing it to a blind person. 
        The description in your response will be sent to an image generation API."""
        
        async with message.channel.typing():
            image_prompt = await create_image_prompt(llm_prompt)
            await pic(ctx, prompt=image_prompt)
            return
    
    if not a1111_online():
        info_embed.title = f"A1111 api is not running at {A1111}"
        info_embed.description = "Launch Automatic1111 with the `--api` commandline argument\nRead more [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)"
        await ctx.reply(embed=info_embed)
    
    if client.behavior.bot_should_reply(message):
        pass # Bot replies.
    else:
        return # Bot does not reply to this message.    
    
    user_input = LLMUserInputs().settings
    user_input["text"] = text
    user_input["state"]["name1"] = message.author.display_name
    user_input["state"]["name2"] = client.user.display_name
    user_input["state"]["context"] = client.llm_context
    num = check_num_in_queue(message)
    if num >=10:
        await message.channel.send(f'{message.author.mention} You have 10 items in queue, please allow your requests to finish before adding more to the queue.')
    else:
        queue(message, user_input)
        async with message.channel.typing():
            await llm_gen(message, queues)

@client.hybrid_command(description="Set current channel as main channel for bot to auto reply in without needing to be called")
async def main(ctx):
    ctx.bot.behavior.main_channels.append(ctx.message.channel.id)
    conn = sqlite3.connect('bot.db')
    c = conn.cursor()
    #c.execute('''INSERT OR REPLACE INTO config (setting, value) VALUES (?, ?)''', ('main_channels', f'{ctx.message.channel.id}'))
    c.execute('''INSERT OR REPLACE INTO main_channels (channel_id) VALUES (?)''', (ctx.message.channel.id,))
    conn.commit()
    conn.close()
    await ctx.reply(f'Bot main channel set to {ctx.message.channel.mention}')
    #await ctx.message.channel.send(f'Bot main channel set to {ctx.message.channel.mention}')

@client.hybrid_command(description="Display help menu")
async def helpmenu(ctx):
    await ctx.send(embed=info_embed)



@client.hybrid_command(description="Take a picture!")
@app_commands.describe(prompt="The initial prompt to contextualize LLaMA")
async def pic(ctx, prompt=None):
    if not a1111_online():
        info_embed.title = f"A1111 api is not running at {A1111}"
        info_embed.description = "Launch Automatic1111 with the `--api` commandline argument\nRead more [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)"
        await ctx.reply(embed=info_embed)
    else:
        info_embed.title = "Processing ..."
        info_embed.description = " "
        picture_frame = await ctx.reply(embed=info_embed)
        if not prompt:
            llm_prompt = """Describe the scene as if it were a picture to a blind person,
            also describe yourself and refer to yourself in the third person if the picture is of you.
            Include as much detail as you can."""
            image_prompt = await create_image_prompt(llm_prompt)
        else:
            image_prompt = prompt
        info_embed.title = "Generating image ..."
        #info_embed.description = image_prompt
        await picture_frame.edit(embed=info_embed)
        payload = { "prompt": image_prompt, "width": 768, "height": 512, "steps": 20, "restore_faces": True } 

        # Looking for SD prompts in the character files
        filepath = next(Path("characters").glob(f"{client.user.display_name}.{{yml,yaml,json}}"), None)
        for extension in ["yml", "yaml", "json"]:
            filepath = Path(f'characters/{client.user.display_name}.{extension}')
            if filepath.exists():
                break
        if filepath:
            with open(filepath) as f:
                data = json.load(f) if filepath.suffix == ".json" else yaml.safe_load(f)
                restore_faces = data.get("restore_faces")
                positive_prompt_prefix = data.get("positive_prompt_prefix")
                positive_prompt_suffix = data.get("positive_prompt_suffix")
                negative_prompt = data.get("negative_prompt")
                presets = data.get("presets")
            if 'selfie' in payload["prompt"]: 
                payload["width"] = 512
                payload["height"] = 768
            if 'instagram' in payload["prompt"]: 
                payload["width"] = 512
                payload["height"] = 512
            if restore_faces == False:
                payload["restore_faces"] = False
            if positive_prompt_prefix: 
                payload["prompt"] = f'{positive_prompt_prefix} {image_prompt}'
            if positive_prompt_suffix:
                payload["prompt"] += " " + positive_prompt_suffix
            if negative_prompt: payload["negative_prompt"] = negative_prompt
            if presets:
                for preset in presets:
                    if preset['trigger'] in payload["prompt"]:
                        payload["prompt"] += " " + preset['positive_prompt']
                        payload["negative_prompt"] += " " + preset['negative_prompt']


        task = asyncio.ensure_future(generate_image_with_a1111(payload))
        try:
            await asyncio.wait_for(task, timeout=25)
        except asyncio.TimeoutError:
            info_embed.title = "Image complete"
            await picture_frame.edit(delete_after=15)
        else:
            file = discord.File(os.path.join(os.path.dirname(__file__), f'{client.user.display_name}.png'))
            info_embed.title = "Image complete"
            #await picture_frame.edit(embed=info_embed, delete_after=5)
            await picture_frame.edit(delete_after=5)
            #info_embed.description = image_prompt
            #info_embed.set_image(url=f"attachment://{client.user.display_name}.png")
            #await picture_frame.delete()
            await ctx.send(file=file)
            await ctx.send(image_prompt)

@client.hybrid_command(description="Reset the conversation with LLaMA")
@app_commands.describe(
    prompt_new="The initial prompt to contextualize LLaMA",
    your_name_new="The name which all users speak as",
    llamas_name_new="The name which LLaMA speaks as")
async def reset(ctx, prompt_new=prompt, your_name_new=your_name, llamas_name_new=llamas_name):
    global reply_count
    your_name = ctx.message.author.display_name
    llamas_name = ctx.bot.user.display_name
    reply_count = 0
    shared.stop_everything = True
    clear_chat_log("", "cai-chat")
    await change_profile(ctx, llamas_name)
    prompt = ctx.bot.llm_context
    info_embed.title = f"Conversation with {llamas_name} reset"
    info_embed.description = ""
    await ctx.reply(embed=info_embed)    
    logging.info("conversation reset: {'replies': " + str(reply_count) + ", 'your_name': '" + your_name + "', 'llamas_name': '" + llamas_name + "', 'prompt': '" + prompt + "'}")
    #reset_embed.timestamp = datetime.now() - timedelta(hours=3)
    #reset_embed.description = "Replies: " + str(reply_count) + "\nYour name: " + your_name + "\nLLaMA's name: " + llamas_name + "\nPrompt: " + prompt

@client.hybrid_command(description="Check the status of your reply queue position and wait time")
async def status(ctx):
    total_num_queued_jobs = len(queues)
    que_user_ids = [list(a.keys())[0] for a in queues]
    if ctx.message.author.mention in que_user_ids:
        user_position = que_user_ids.index(ctx.message.author.mention) + 1
        msg = f"{ctx.message.author.mention} Your job is currently {user_position} out of {total_num_queued_jobs} in the queue. Estimated time until response is ready: {user_position * 20/60} minutes."
    else:
        msg = f"{ctx.message.author.mention} doesn\'t have a job queued."

    status_embed.timestamp = datetime.now() - timedelta(hours=3)
    status_embed.description = msg
    
    await ctx.send(embed=status_embed)

def generate_characters():
    cards = []
    # Iterate through files in image folder
    for file in sorted(Path("characters").glob("*")):
        if file.suffix in [".json", ".yml", ".yaml"]:
            character = {}
            character["name"] = file.stem
            filepath = str(Path(file).absolute())
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f) if file.suffix == ".json" else yaml.safe_load(f)
                description = data.get("bot_description")
                emoji = data.get("bot_emoji")
                #custom emojis are like this <:sheila:576121845426814986> you get it by doing \:sheila:
                character["bot_description"] = description if description else None
                character["bot_emoji"] = emoji if emoji else "💬" #🧠
                cards.append(character)
    return cards

class Dropdown(discord.ui.Select):
    def __init__(self, ctx):
        options = [discord.SelectOption(label=character["name"], description=character["bot_description"], emoji=character["bot_emoji"]) for character in generate_characters()]
        super().__init__(placeholder='', min_values=1, max_values=1, options=options)
        self.ctx = ctx

    async def callback(self, interaction: discord.Interaction):
        character = self.values[0]
        #await interaction.response.send_message(f'Selection: {character}')
        await change_profile(self.ctx, character)
        if self.view:
            # Trying desperately to remove the dropdown menu after use, but none of these are working
            self.view.stop()
            self.view.is_finished() 
            self.view.clear_items()

@client.hybrid_command(description="Choose Character")
@commands.cooldown(1, 600, commands.BucketType.guild)
@app_commands.describe()
async def character(ctx):
    view = DropdownView(ctx)
    if hasattr(ctx.bot, "last_change"):
        if datetime.now() >= ctx.bot.last_change + timedelta(minutes=10):
            remaining_cooldown = ctx.bot.last_change + timedelta(minutes=10) - datetime.now() 
            remaining_cooldown = total_seconds = remaining_cooldown.total_seconds()
            await ctx.channel.send(f'`Please wait {ceil_timedelta(remaining_cooldown)} before changing character again`')
    else:
        await ctx.send('Choose Character:', view=view)

class DropdownView(discord.ui.View):
    def __init__(self, ctx):
        super().__init__()
        self.add_item(Dropdown(ctx))

class LLMUserInputs():
    def __init__(self):
        self.settings = {
        "text": "",
        "state": {
            "max_new_tokens": 400,
            "seed": -1.0,
            "temperature": 0.7,
            "top_p": 0.1,
            "top_k": 40,
            "typical_p": 1,
            "repetition_penalty": 1.18,
            "encoder_repetition_penalty": 1,
            "no_repeat_ngram_size": 0,
            "min_length": 50,
            "do_sample": True,
            "penalty_alpha": 0,
            "num_beams": 1,
            "length_penalty": 1,
            "early_stopping": False,
            "add_bos_token": True,
            "ban_eos_token": False, 
            "skip_special_tokens": True,
            "truncation_length": 2048,
            "custom_stopping_strings": '"### Assistant","### Human","</END>"',
            "name1": "",
            "name2": client.user.display_name,
            "name1_instruct": "",
            "name2_instruct": client.user.display_name,
            "greeting": "",
            "context": client.llm_context,
            "end_of_turn": "",
            "chat_prompt_size": 2048,
            "chat_generation_attempts": 1,
            "stop_at_newline": False,
            "mode": "cai-chat",
            "stream": True
        },
        "regenerate": False,
        "_continue": False
    } 

class Behavior():
    def __init__(self):
        """ Settings for the bot's behavior. Intended to be accessed via a command in the future """
        self.learn_about_and_use_guild_emojis = None # Considering a specific command that randomly asks about one unknown emoji
        self.take_notes_about_users = None # Will consume tokens to loop this back into the context but could be worth it to fake a long term memory
        self.read_chatlog = None # Feed a few lines on character change from the previous chat session into context to make characters aware of each other.
        """ Those above are not yet implemented and possibly terrible ideas """
        self.reply_with_image = 0 # Chance for the bot to respond with an image instead of just text
        self.change_username_with_character = True
        self.change_avatar_with_character = True
        self.only_speak_when_spoken_to = True
        self.ignore_parenthesis = True
        self.reply_to_itself = 0
        self.chance_to_reply_to_other_bots = 0.3 #Reduce this if bot is too chatty with other bots
        self.reply_to_bots_when_adressed = 0.5 #random.random()
        self.go_wild_in_channel = True 
        self.user_conversations = {} # user ids and the last time they spoke.
        self.conversation_recency = 600
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS emojis (emoji TEXT UNIQUE, meaning TEXT)''')    # set up command for bot to ask and learn about emojis
        c.execute('''CREATE TABLE IF NOT EXISTS config (setting TEXT UNIQUE, value TEXT)''')    # stores settings
        c.execute('''CREATE TABLE IF NOT EXISTS main_channels (channel_id TEXT UNIQUE)''')      # new separate table for main_channels 
        #c.execute('''CREATE TABLE IF NOT EXISTS usernotes (users, message, notes, keywords)''')
        c.execute('''SELECT channel_id FROM main_channels''')
        result = c.fetchall()
        print(result)
        if result is not []:
            self.main_channels = result
        else:
            self.main_channels = None
        conn.commit()
        conn.close()
    
    def update_user_dict(self, user_id):
        self.user_conversations[user_id] = datetime.now()
    
    def in_active_conversation(self, user_id):
        if user_id in self.user_conversations:
            last_conversation_time = self.user_conversations[user_id]
            time_since_last_conversation = datetime.now() - last_conversation_time
            if time_since_last_conversation.total_seconds() < self.conversation_recency:
                #logging.info(f'behavior: {user_id} is in active conversation')
                return True
            else:
                return False
        else:
            return False

    def bot_should_reply(self, message):
        """ Beware spaghetti ahead """
        reply = False
        if message.author == client.user: 
            return False
        if message.author.bot and client.user.display_name.lower() in message.clean_content.lower() and message.channel.id in self.main_channels:
            """ if using this bot's name in the main channel and another bot is speaking """
            reply = self.probability_to_reply(self.reply_to_bots_when_adressed)
            #logging.info(f'behavior: reply_to_bots_when_adressed triggered {reply=}')
            if 'bye' in message.clean_content.lower():
                """ if other bot is trying to say goodbye, just stop replying so it doesn't get awkward """
                return False
            
        if self.ignore_parenthesis and \
            (message.content.startswith('(') and message.content.endswith(')') \
            or \
            (message.content.startswith('<:') and message.content.endswith(':>'))):
            """ if someone is simply using an <:emoji:> or (speaking like this) """
            return False
        
        if (self.only_speak_when_spoken_to and client.user.mentioned_in(message) \
                    or any(word in message.content.lower() for word in client.user.display_name.lower().split())) \
                or (self.in_active_conversation(message.author.id) and message.channel.id in self.main_channels):
            """ If bot is set to only speak when spoken to and someone uses its name
                or if is in an active conversation with the user in the main channel, we reply. 
                This is a messy one. """
            #logging.info(f'behavior: only_speak_when_spoken_to triggered')
            return True
        else:
            reply = False 
            #logging.info(f'behavior: only_speak_when_spoken_to triggered {reply=}')

        if message.author.bot and message.channel.id in self.main_channels: 
            reply = self.probability_to_reply(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and message.channel.id in self.main_channels: 
            reply = True
            #logging.info(f'behavior: go_wild_in_channel {reply=}')
        if reply == True: 
            self.update_user_dict(message.author.id)
            #logging.info(f'behavior: {reply=}')
        return reply

    def probability_to_reply(self, probability):
        """ 1 always returns True. 0 always returns False. 0.5 has 50% chance of returning True. """
        roll = random.random()
        return roll < probability

def queue(message, user_input):
    user_id = message.author.mention
    queues.append({user_id:user_input})
    logging.info(f'reply requested: "{user_id} asks {user_input["state"]["name2"]}: {user_input["text"]}"')

def check_num_in_queue(message):
    user = message.author.mention
    user_list_in_que = [list(i.keys())[0] for i in queues]
    return user_list_in_que.count(user)

async def generate_image_with_a1111(payload):
    print(payload)
    response = requests.post(url=f'{A1111}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{A1111}/sdapi/v1/png-info', json=png_payload)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(f'{client.user.display_name}.png', pnginfo=pnginfo)
    return image

async def get_progress():
    url = "http://127.0.0.1:7860/sdapi/v1/progress"
    response = await loop.run_in_executor(None, requests.get, url)
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print("Error:", response.status_code)

if not hasattr(client, 'behavior'):
    client.behavior = Behavior()    
client.run(bot_args.token if bot_args.token else TOKEN, root_logger=True)
