# ChatLLaMA Discord Bot

A Discord Bot for chatting with LLaMA, Vicuna, Alpaca, or any other LLaMA-based model. It's not as good as ChatGPT but LLaMA and its derivatives are pretty impressive on their own. Tag the bot with it's `@username` or mention it by name to talk to it. Give it a channel of its own to avoid having to tag it every time. Use `/character` to change to other characters in your character folder. To clear chat history with LLaMA or change the initial prompt, use `/reset`. Sometimes LLaMA will get stuck or you will want to change the initial prompt to something more interesting so `/reset` is well used.

Can use A1111 if it is in api mode with `/pics` command: 

![image](https://user-images.githubusercontent.com/37743453/235309643-316b0f68-58d6-4023-bb4b-d86d2a212ce5.png)

Can also just ask directly to `take a selfie` or `take a picture`:

![image](https://user-images.githubusercontent.com/37743453/235515685-4b80770c-509e-4afa-8cb1-5b539b6bc578.png)
![image](https://user-images.githubusercontent.com/37743453/235619741-b7eb9c47-860f-4d08-ad99-3ef959d82241.png)

Note the additions to the Llayla character in the characters folder, showing how to provide optional SD prompts the bot can use when taking pictures or selfies, if you're after a particular style or look. It will use whatever model is loaded in the webgui of Automatic 1111.


# Setup

1. Setup text-generation-webui with their [one-click installer](https://github.com/oobabooga/text-generation-webui#one-click-installers) and download the model you want (for example `decapoda-research/llama-7b-hf`). Make sure it's working.

2. Edit `bot.py` with your Discord bot's token

3. Place `bot.py` inside the text-generation-webui directory

4. Open the `cmd` file that came with the one-click installer

5. Install discord.py with `pip install discord`

6. Run the bot with `python bot.py`

Note: For ease of use `bot.py` supports all `server.py` model-related command line arguments. For example, `python bot.py --model llama-7b-hf` is a valid way to start the bot.

# Example Transcript

```
Your name is James and you love having conversations. You write long sentences and write very eloquently. Your responses are many sentences long. You enjoy talking with people and engaging them in interesting topics. My name is Robert and I like to have conversations with you James.

Robert: Hi James, how are you doing today?
James:  Hello Robert, I am doing fine today.

Robert: I have heard you are an amazing person! I am so glad to finally meet you!
James:  You are so kind! I am happy to finally meet you too.

Robert: How is Sarah and the kids? Is the kid's new school turning out as good as you had hoped?
James:  Sarah is doing great, and our children are growing up so quickly! The new school is a lot better than I had expected.

Robert: Thank goodness! I remember in my day, we had 3 classrooms and only one chalkboard! Schools around here have really grown!
James:  I remember those days. I think the world is getting better and better. I like that!

Robert: Do you remember that time in astronomy when John saw a shooting star?! He cataloged it and it turned out to be a new comet! I saw a news story on it recently. Apparently, they're going to name it after him! They want to use his last name though, because his first name is too generic
James:  I remember that. I am happy for John. That is awesome news.

Robert: Quick question, what is the circumference of the Earth? It was a question on Lindsey's homework last night and for the life of me, I just can't remember the answer!
James:  Hmm, I do not know that answer. I will look it up.

Robert: Sounds good! What did you find?
James:  The circumference of the Earth is 24,901 miles.

Robert: Ahhh very good to know. These days, the world feels like it's getting smaller and smaller with the internet!
James:  Indeed! It is crazy to think about how the world has grown to be so small in recent years.
```
