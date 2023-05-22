note: last working ooba commit: `a5d5bb9`

# ChatLLaMA Discord Bot

A Discord Bot for chatting with LLaMA, Vicuna, Alpaca, or any other LLaMA-based model. It's not as good as ChatGPT but LLaMA and its derivatives are pretty impressive on their own. Tag the bot with it's `@username` or mention it by name to talk to it. Give it a channel of its own to avoid having to tag it every time. Use `/character` to change to other characters in your character folder. 

Use `regen` to make the bot forget and regenerate the previous response. Useful to guide it in a direction. Use `/cont` if you're too lazy to type "continue".

To clear chat history with LLaMA or change the initial prompt, use `/reset`. Sometimes LLaMA will get stuck or you will want to change the initial prompt to something more interesting so `/reset` is well used.

Can use A1111 if it is in api mode with `/pics` command: 

![image](https://user-images.githubusercontent.com/37743453/235309643-316b0f68-58d6-4023-bb4b-d86d2a212ce5.png)

Can also just ask directly to `take a selfie` or `take a picture`:

![image](https://user-images.githubusercontent.com/37743453/235515685-4b80770c-509e-4afa-8cb1-5b539b6bc578.png)
![image](https://user-images.githubusercontent.com/37743453/235619741-b7eb9c47-860f-4d08-ad99-3ef959d82241.png)

Note the additions to the Llayla character in the characters folder, showing how to provide optional SD prompts the bot can use when taking pictures or selfies, if you're after a particular style or look. It will use whatever model is loaded in the webgui of Automatic 1111.


# Setup

1. Setup text-generation-webui with their [one-click installer](https://github.com/oobabooga/text-generation-webui#one-click-installers) and download the model you want (for example `decapoda-research/llama-7b-hf`). Make sure it's working.

2. Edit `config.py` with your Discord bot's token

3. Place `bot.py` and `config.py` inside the text-generation-webui directory

4. Place your character file(s) in the `characters` folder. One of them should have the same name as your bot.

5. Open the `cmd` file that came with the one-click installer

6. Install discord.py with `pip install discord`

7. Run the bot with `python bot.py`

# Notes: 
- For ease of use `bot.py` supports all `server.py` model-related command line arguments. For example, `python bot.py --model llama-7b-hf` is a valid way to start the bot.
- If you experience problems, it might be because the webui updated. Try `git checkout a5d5bb9` before running the bot with `python bot.py`


