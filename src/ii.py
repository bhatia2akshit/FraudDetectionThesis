from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

text='''Dhoom dhoom come and light my fire

Dhoom dhoom let me take you higher

Dhoom dhoom I wanna feel that burnin'

Dhoom dhoom it's a wild emotion


Dhoom dhoom passion and devotion

Dhoom dhoom now the wheels are turnin'

Move your body close to mine now

Let me feel your love divine now


Together we'll explode and we'll go boom boom boom

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

Dhoom dhoom I'm gonna make you sweat now


Dhoom dhoom let's get all wet now

Dhoom dhoom gotta get down on it


Dhoom dhoom till the early morning

Dhoom dhoom until the dawning

Dhoom dhoom I know that you want it

Shake your body down to the ground


Once you on there's no turnin' round

Tonight we're gonna make the world go boom boom boom


Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

(Dance with me, dance with me


This is my philosophy)

(Dance with me, dance with me, oh yeah)


(Dance with me, dance with me

This is my philosophy)





(Dance with me, dance with me, oh yeah)

This burnin' inside


You know you just cannot hide

So don't fight the feeling let your body decide


When you get down on the road

It's a wild overload


Ridin' higher than you ever did before

Doom dhoom let your body do the talkin'

Dhoom dhoom I wanna keep on rockin'

Dhoom dhoom I want it twenty four seven

Dhooom dhoom get the rhythm of the beat now

Dhoom dhoom feel the fire and the heat now

Dhoom dhoom take a trip to heaven

I wanna feel the wind in my hair now

Spread the power, everywhere now

Feel the magic just go zip zap zoom

Dhoom machale. Come on all you people

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom

Dhoom machale Dhoom machale Dhoom'''

filepath = './shahrukh.wav'
# generate speech by cloning a voice using default settings
tts.tts_to_file(text=text,
                file_path='./filepath.wav',
                speaker_wav=filepath,
                language="en")