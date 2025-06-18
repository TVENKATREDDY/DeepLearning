import speech_recognition as sr
import pyttsx3 as pt
import pywhatkit as wh
listening=sr.Recognizer()
engine=pt.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()
def hear():
    cmd=''
    try:
        with sr.Microphone() as mic:
            print('Listening.....')
            voice=listening.listen(mic)
            cmd=listening.recognize_google(voice)
            cmd=cmd.lower()
            print(cmd)
            if 'kodi' in cmd:
                cmd=cmd.replace('kodi','')
                print(cmd)
                
    except sr.UnknownValueError:
        print('sorry, I didnt understad')
    except sr.RequestError:
        print('Sorry I did not understand')
    return cmd
def run():
    cmd=hear()
    if cmd:
        print(f'Command {cmd}')
        if 'play' in cmd:
            song=cmd.replace('play','').strip()
            speak(f'Playing {song}')
            wh.playonyt(song)
        else:
            speak("I didnot cathc that. Could you please repest")
run()            
    
        
                