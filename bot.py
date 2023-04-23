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

### Replace TOKEN with discord bot token
TOKEN = "YOURDISCORDBOTTOKEN"
# Once the bot is online, you can use the /main command to set a channel for it


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

help_embed_json = {
    "title": "Help menu",
    "description": 
      """
      **/character** - Change character \n 
      **/main** - Set main channel for bot so it can reply without being called by name
      """
}
help_embed = discord.Embed().from_dict(help_embed_json)

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
    return (td + timedelta(minutes=1) - timedelta(seconds=td.seconds % 60)).replace(microsecond=0)

async def change_profile(ctx, character):
    """ Changes username and avatar of bot. """
    """ Will be rate limited by discord api if used too often. Needs a cooldown. 10 minute value is arbitrary. """
    #name1, name2, picture, greeting, context, end_of_turn, chat_html_wrapper = load_character(character, '', '', '')
    if hasattr(ctx.bot, "last_change"):
        if datetime.now() >= ctx.bot.last_change + timedelta(minutes=10):
            remaining_cooldown = ctx.bot.last_change + timedelta(minutes=10) - datetime.now() 
            await ctx.channel.send(f'Please wait {ceil_timedelta(remaining_cooldown)} before changing character again')
    else:
        try:
            if (ctx.bot.behavior.change_username_with_character):
                await client.user.edit(username=character)
            if (ctx.bot.behavior.change_avatar_with_character):
                folder = 'characters'
                picture_path = os.path.join(folder, f'{character}.png')
                if os.path.exists(picture_path):
                    with open(picture_path, 'rb') as f:
                        picture = f.read()
                    await client.user.edit(avatar=picture)
            new_char = load_character(character, '', '', '')
            greeting = new_char[3]
            ctx.bot.llm_context = new_char[4]
            #await send_long_message(ctx.channel, greeting)
            file = discord.File(picture_path, filename=f'{character}.png')
            greeting_embed.title=character
            greeting_embed.description=greeting
            #greeting_embed.set_thumbnail(url=f"attachment://{character}.png")
            greeting_embed.set_image(url=f"attachment://{character}.png")
            await ctx.channel.send(file=file, embed=greeting_embed)
            ctx.bot.last_change = datetime.now()
        except discord.HTTPException as e:
            """ This exception can happen when you restart the bot and change character too fast without last_change being set """
            await ctx.channel.send(f'`{e}`')
        except Exception as e:
            print (e)

    if ctx.bot.behavior.read_chatlog:
        """  Allow bot to read recent chatlog somehow. 
        Might want to do this somewhere else. 
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

@client.event
async def on_message(message):
    if client.behavior.bot_should_reply(message):
        pass # Bot replies.
    else:
        return # Bot does not reply to this message.
    async with message.channel.typing():
        text = message.clean_content
        max_new_tokens=200
        do_sample=True
        temperature=0.7
        top_p=0.1
        typical_p=1
        repetition_penalty=1.18
        encoder_repetition_penalty=1
        top_k=40
        min_length=0
        no_repeat_ngram_size=0
        num_beams=1
        penalty_alpha=0
        length_penalty=1
        early_stopping=False
        seed=-1.0
        name1=message.author.display_name
        name2=client.user.display_name
        context=client.llm_context
        stop_at_newline=True
        chat_prompt_size=2048
        chat_generation_attempts=1
        regenerate=False
        mode="cai-chat"
        end_of_turn=""
        add_bos_token=True
        custom_stopping_string=""
        _continue=False
        user_input = {
            "text": text,
            "state": {
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "typical_p": typical_p,
                "repetition_penalty": repetition_penalty,
                "encoder_repetition_penalty": encoder_repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "min_length": min_length,
                "do_sample": do_sample,
                "penalty_alpha": penalty_alpha,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "early_stopping": early_stopping,
                "add_bos_token": add_bos_token,
                "ban_eos_token": False,
                "skip_special_tokens": True,
                "truncation_length": 2048,
                "custom_stopping_strings": custom_stopping_string,
                "name1": name1,
                "name2": name2,
                "greeting": "",
                "context": context,
                "end_of_turn": end_of_turn,
                "chat_prompt_size": chat_prompt_size,
                "chat_generation_attempts": chat_generation_attempts,
                "stop_at_newline": stop_at_newline,
                "mode": mode
            },
            "regenerate": regenerate,
            "_continue": _continue
        }

        num = check_num_in_queue(message)
        if num >=10:
            await message.channel.send(f'{message.author.mention} You have 10 items in queue, please allow your requests to finish before adding more to the queue.')
        else:
            queue(message, user_input)
            await llm_gen(message, queues)

@client.hybrid_command(description="Set current channel as main channel for bot to auto reply in without needing to be called")
async def main(ctx):
    ctx.bot.behavior.main_channel = ctx.message.channel.id
    await ctx.respond(f'Bot main channel set to {ctx.message.channel.mention}')
    #await ctx.message.channel.send(f'Bot main channel set to {ctx.message.channel.mention}')

@client.hybrid_command(description="Display help menu")
async def helpmenu(ctx):
    await ctx.send(embed=help_embed)

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
    clear_chat_log(your_name, llamas_name, "", "")
    await change_profile(ctx, llamas_name)
    prompt = ctx.bot.llm_context
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
            character = file.stem
            cards.append(character)
    # Nabbed the changes suggested by Hárold
    # Maybe look for descriptions and emojis as well? 
    # discord.SelectOption( label="Llayla", description="Assistant", emoji='🟣'), 
    return cards

class Dropdown(discord.ui.Select):
    def __init__(self, ctx):
        options = [discord.SelectOption(label=name) for name in generate_characters()]
        super().__init__(placeholder='', min_values=1, max_values=1, options=options)
        self.ctx = ctx

    async def callback(self, interaction: discord.Interaction):
        character = self.values[0]
        await interaction.response.send_message(f'Selection: {character}')
        self.disabled = True  # Supposed to hide dropdown after use. Doesn't seem to do the job.
        await change_profile(self.ctx, character)

@client.hybrid_command(description="Choose Character")
@commands.cooldown(1, 60, commands.BucketType.guild)
@app_commands.describe()
async def character(ctx):
    view = DropdownView(ctx)
    await ctx.send('Choose Character:', view=view)

class DropdownView(discord.ui.View):
    def __init__(self, ctx):
        super().__init__()
        self.add_item(Dropdown(ctx))

class Behavior():
    def __init__(self):
        """ Settings for the bot's behavior. Intended to be accessed via a /command in the future """
        self.learn_about_and_use_guild_emojis = None # Will consume tokens
        self.take_notes_about_users = None # Will consume tokens
        self.read_chatlog = None # Feed a few lines on character change from the previous chat session into context to make characters aware of each other.
        """ Those with None are not yet implemented and possibly terrible ideas """
        self.change_username_with_character = True
        self.change_avatar_with_character = True
        self.main_channel = 123 # Why not set the channel in here via bot commands instead of hardcoding like a maniac.
        self.only_speak_when_spoken_to = True
        self.ignore_parenthesis = True
        self.reply_to_itself = 0
        self.chance_to_reply_to_other_bots = 0.3 #Reduce this if bot is too chatty with other bots
        self.reply_to_bots_when_adressed = random.random()
        self.go_wild_in_channel = True
        self.user_conversations = {} # user ids and the last time they spoke.
        self.conversation_recency = 600

        import sqlite3
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS emojis (emoji, meaning)''') # set up command for bot to ask and learn about emojis
        c.execute('''CREATE TABLE IF NOT EXISTS config (setting, value)''') # future long term storage for main channel etc
        #c.execute('''CREATE TABLE IF NOT EXISTS usernotes (users, message, notes, keywords)''')
        conn.commit()
        conn.close()
    
    def update_user_dict(self, user_id):
        self.user_conversations[user_id] = datetime.now()
    
    def in_active_conversation(self, user_id):
        if user_id in self.user_conversations:
            last_conversation_time = self.user_conversations[user_id]
            time_since_last_conversation = datetime.now() - last_conversation_time
            if time_since_last_conversation.total_seconds() < self.conversation_recency:
                return True
            else:
                return False
        else:
            return False

    def bot_should_reply(self, message):
        reply = False
        if message.author.bot and client.user.display_name.lower() in message.clean_content.lower() and message.channel.id == self.main_channel:
            """ If another bot is speaking and using this bot's name in the main channel """
            reply = self.probability_to_reply(self.reply_to_bots_when_adressed)
            if 'bye' in message.clean_content.lower():
                """ if other bot is trying to say goodbye, just stop replying so it doesn't get awkward """
                return False
            
        if self.ignore_parenthesis and \
            (message.content.startswith('(') and message.content.endswith(')') \
            or \
            (message.content.startswith(':') and message.content.endswith(':'))):
            """ if someone is simply using an :emoji: or (speaking like this) """
            return False
        
        if (self.only_speak_when_spoken_to and client.user.mentioned_in(message) \
                    or any(word in message.content.lower() for word in client.user.display_name.lower().split())) \
                or (self.in_active_conversation(message.author.id) and message.channel.id == self.main_channel):
            """ If bot is set to only speak when spoken to and someone uses its name
                or if is in an active conversation with the user in the main channel, we reply. """
            reply = True

        if message.author.bot and message.channel.id == self.main_channel: reply = self.probability_to_reply(self.chance_to_reply_to_other_bots)
        if self.go_wild_in_channel and message.channel.id == self.main_channel: reply = True
        if message.author == client.user: reply = self.probability_to_reply(self.reply_to_itself)
        if reply == True: self.update_user_dict(message.author.id)
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


client.run(bot_args.token if bot_args.token else TOKEN, root_logger=True)
