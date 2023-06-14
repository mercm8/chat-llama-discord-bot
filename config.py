discord = {'TOKEN': "YOURDISCORDTOKENHERE" }

sd = {
    'A1111' : "http://127.0.0.1:7860", #This is the default URL for the A1111 API. If you don't have one, dont worry about it.
    'payload' : {
        'restore_faces': True,
        'sampler_name': "DPM++ 2M Karras",
        'steps': 20,
        'cfg_scale': 5,
        'enable_hr': False,
        'hr_upscaler': "4x-UltraSharp",
        'denoising_strength': 0.55,
        'hr_scale': 2,
        'hr_second_pass_steps': 20
        }
    }

llm = {
    'state': {
            'max_new_tokens': 400,
            'seed': -1.0,
            'temperature': 0.7,
            'top_p': 0.1,
            'top_k': 40,
            'typical_p': 1,
            'repetition_penalty': 1.18,
            'encoder_repetition_penalty': 1,
            'no_repeat_ngram_size': 0,
            'min_length': 50,
            'do_sample': True,
            'penalty_alpha': 0,
            'num_beams': 1,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'add_bos_token': True,
            'ban_eos_token': False, 
            'skip_special_tokens': True,
            'truncation_length': 2048,
            'custom_stopping_strings': '"### Assistant","### Human","</END>"',
            'greeting': "",
            'end_of_turn': "",
            'chat_prompt_size': 2048,
            'chat_generation_attempts': 1,
            'stop_at_newline': False,
            'mode': "cai-chat",
            'stream': True
        }
        }

behavior = {
    # Numbers indicate a chance. 0 never happens. 1 always happens.
    'reply_with_image' : 0, 
    'change_username_with_character' : True,
    'change_avatar_with_character' : True,
    'only_speak_when_spoken_to' : True,
    'ignore_parenthesis' : True,
    'reply_to_itself' : 0,
    'chance_to_reply_to_other_bots' : 0.3, #Reduce this if bot is too chatty with other bots
    'reply_to_bots_when_adressed' : 0.5, # If set to 1, bots can be stuck in an infinite conversation
    'go_wild_in_channel' : True, 
    'conversation_recency' : 600}
