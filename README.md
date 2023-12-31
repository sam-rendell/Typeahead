# Failed Experiment / LTSM Keyboard Dynamics

Predicting keystrokes from timing intervals using LSTM.
-  Around 45% validation accuracy
- Timing intervals are a bit of a jumble
- Model fit - square peg in a round hole. 
- Needs shuffle in architecture
- Increased loss / variance hitting TCP stack (QUIC quicker?)
## References
- Timing Analysis of Keystrokes and Timing Attacks on SSH - https://people.eecs.berkeley.edu/~daw/papers/ssh-use01.pdf
- Feasibility of a Keystroke Timing Attack on Search Engines with Autocomplete - https://ieeexplore.ieee.org/document/8844619