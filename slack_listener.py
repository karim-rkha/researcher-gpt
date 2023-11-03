import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from dotenv import load_dotenv

load_dotenv()

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ["DOTHIS_BOT_TOKEN"])

# #Message handler for Slack
# @app.message(".*")
# def message_handler(message, say, logger):
#     print(message)
     
#     say("Here here")

@app.event("app_mention")
def handle_app_mention_events(body, logger):
    user_prompt = body["event"]["blocks"][0]["elements"][0]["elements"][1]["text"]
    channel_id = body["event"]["channel"]
    os.environ["SLACK_CHANNEL_ID"] = str(channel_id)
    print(user_prompt)
    cmd = f'python app_bis.py "{user_prompt}"'
    # cmd = f'python ../run.py --task "{user_prompt}" --name "bricks_game"'
    print(cmd)
    os.system(cmd)
    

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["DOTHIS_APP_TOKEN"]).start()