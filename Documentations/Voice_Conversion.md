# Voice Conversion
Voice Conversion (VC) is to convert a speech audio file into an audio file of another person’s voice. There are numerous different ways to perform voice conversion, some of which are chosen to generated spoofed data for this project.


## Conversion Data
VC spoofed data are generated using machine learning or algorithm models trained using bona fide data that is provided by the IMDA. The following specified speakers are chosen for generating the data. 
SPEAKER0001 SPEAKER0002 SPEAKER0004 SPEAKER0011 SPEAKER0017 (female)
SPEAKER0006 SPEAKER0007 SPEAKER0008 SPEAKER0009 SPEAKER0010 (male)

SPEAKER0001, SPEAKER0002, SPEAKER0006, SPEAKER0007 will be used as target speech, 
while SPEAKER0098, SPEAKER0099, SPEAKER0100, SPEAKER0101 (male, male, female, female) will be used as target speech

100 of each target speech will be used for generating spoofed VC data for each VC models, thus each models will be generating 800 data, resulting in 8000 spoofed data.

## Mangio-RVC
[Mangio RVC](https://github.com/Mangio621/Mangio-RVC-Fork) is a fork from the original RVC repository with added feature of a different web interface, compatibility with hugging face and paperspace and more. [Retrival Voice Conversion (RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md) is a easy to use voice conversion framework based on [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits). 

### VITS
VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech is a leading voice conversion algorithm. It adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. More details can be found in the [model's git repository](https://github.com/jaywalnut310/vits). 

### Training VC Model
Training of VC model can be done using the web-based UI interface, instructions can be found in [Mangio's git repository](https://github.com/Mangio621/Mangio-RVC-Fork). The setting and specification of the training are as follows, Target Sample Rate: 48k, using pitch guidance, version 2, rmvpe pitch extraction algorithm, 500 training epoch and 24 batch size for best results. Pretrained model f0D40k.pth and f0G40k.pth were used to speed up training and improve training result.

### Generation of spoofed data (Inference)
All the data generated here are using the rmvps pitch extraction algorithm, with value 7 for applying median filtering to harvested pitch results. The pitch adjustment varies for different speaker, but the other parameters are at default constant of 0 resampling, 1 volume envelop of the input and 0.33 protection voiceless consonants.

To change the pitch of the audio, the audio transpose value can be changed. It is a manual process of changing, however a procedure can be utillized to process large amount of VC model faster. The method would require a batch of VC models and a set batch of target speeches.

eg. VC model: 0001, 0002, 0003, 0004. Target speech voice: 0005, 0006, 0007, 0008.

Choosing one representative VC model and do manual pitch adjustment to match generated target speech voices to sound as close as possible to original voice from VC model. Record the transpose value down.

eg. For VC model 1, target speech voice to transpose value: (0005, -4), (0006, 3), (0007, 10), (0007, -15)

For all other VC models, do manual pitch adjustment the same way only for the first target speech voice. The remaining target speech voice can be calculated relative to the first target speech voice.

eg. For VC model 2, target speech voice 0005 has transpose value of -1. Compared to VC model 1's target speech voice 0005 transpose value, VC model 2 has transpose value of 3 increment, -1-(-4) = 3. Thus all other transpose value would likely be 3 increment from VC model 1. <br>
For VC model 1, target speech voice to transpose value: (0005, -4+3), (0006, 3+3), (0007, 10+3), (0007, -15+3)

Further manual testing can be done to double check the pitch.

