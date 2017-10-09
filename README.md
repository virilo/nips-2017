# Last day solution for the NIPS-2017 non targeted attack competition.


I'd have liked to do this approach: https://github.com/virilo/nips-2017-Carlini-Wagner-non-targeted-adversarial-attack
But it wasn't possible during the competition: https://stackoverflow.com/questions/46502291/error-getting-inceptionv3-logits-in-tensorflow
And I ended up doing this solution the last day of the competition :-/

The last day idea was based on the concept that attacks on one model are supposed to have success against other Machine Learning or Deep Learning models trained with the same data (black-box attacks)
I attacked the same image using different CNNs (all of them were trained with the same data)
Then I did a weighted average of the adversarial images.
Weights were found manually, using the most difficult images to be attacked as dataset.

Thanks the organizers of this interesting competition and to all researchers investigating about this attacks.


