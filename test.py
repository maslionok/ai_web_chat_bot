from pyngrok import ngrok

# (only need to do this once per machine)
ngrok.set_auth_token("2wwlddQaYc1aNiXFK85Ikg89WBV_5DToEmHadBpyUxfBeyrN3")

# open a tunnel to your local port 5000
public_url = ngrok.connect(5000)
print(" * ngrok tunnel:", public_url)
