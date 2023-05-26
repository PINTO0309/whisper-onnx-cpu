import pyaudio
import speech_recognition as sr

# create recognizer and mic instances
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16_000, device_index=10)

try:
    while True:  # infinite loop to keep listening
        with mic as audio_source:
            print("Speak now!")
            # adjust recognizer sensitivity to ambient noise and record audio from the mic
            recognizer.adjust_for_ambient_noise(audio_source)
            audio = recognizer.listen(audio_source)

        try:
            # recognize speech using Google Speech Recognition
            print("You said: " + recognizer.recognize_google(audio, language='ja'))

        except sr.UnknownValueError:
            # Google Speech Recognition could not understand audio
            print("Google Speech Recognition could not understand the audio")

        except sr.RequestError as e:
            # could not request results from Google Speech Recognition service
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

except KeyboardInterrupt:
    pass  # allow CTRL + C to exit the application