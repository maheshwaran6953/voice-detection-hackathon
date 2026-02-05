import base64

# Read your audio file
with open("trial.wav", "rb") as audio_file:
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

print(audio_base64[:100] + "...")  # Print first 100 chars to verify
print("\nFull Base64 string is ready!")

# Optionally save to a file
with open("audio_base64.txt", "w") as f:
    f.write(audio_base64)