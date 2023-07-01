# Generating Music with GPT
Coding up a GPT transformer from scratch, training to generate midi music.

In this article, I detail my learnings implementing a GPT model from scratch, training it for a custom use case - generating symbolic music.

## Results
[![Embedded YouTube Video](https://img.youtube.com/vi/kLn-hvynM3I/0.jpg)](https://youtu.be/kLn-hvynM3I)

I deployed the GPT model in flask and modified a cool midi player project I found on github (https://github.com/ryohey/signal) to complete input prompt music sequences. As evident from the video, they were not all winners 😜- some generations were horrifyingly repulsive. But in most cases, the model does a decent job.


## GPT from scratch
Below is a diagram showing the Transformer -decorder architecture I implemented.

![Image 1](https://github.com/kvsnoufal/MidiTransformer/blob/main/docs/transformers.drawio.png)
GPT(Transformer-decoder architecture) - 2 Heads, 1 Layer.


![Image 2](https://github.com/kvsnoufal/MidiTransformer/blob/main/docs/img2.png)
Tiny Stories completion.


