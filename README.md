# Using Amortized Bayesian Inference to Simulate Learning of Chord Types in Music

Interacting with music requires us to make inferences on several levels. When listening to a piece of music, we infer its structure. After listening to many pieces, we infer general regularities, i.e. we learn. Bayesian inference is a useful formalism for making these inferences in a principled way, and is usedto model inferences in cognition in general. However, exact Bayesian inference it is also known tobe intractable in the general case so approximate inference methods such as MCMC and variational inference are needed. This project applies Bayesian inference to the problem of learning chord types and their properties. The project builds on an existing model for the supervised case that is implemented as a probabilistic program using variational inference. In the unsupervised case, however, standard variational inference fails. The goal of this project is to solve the unsupervisedinference problem using amortized inference, which is an extension of variational inference that uses deep learning.

This thesis explores the application of amortized Bayesian inference to learn and identify chord types from unlabeled musical data. Music with its complex structures and variety makes accurate probabilistic modeling difficult. Traditional methods like variational inference (VI) often fail in unsupervised settings because of the complexity and lack of labels in musical data. This study applied amortized inference, which incorporated deep learning to approximate the posterior distribution. The thesis is based on an existing model designed for chordtones and ornaments recognition using Bayesian modeling. A simplified prototype that does not distinguish between chordtones and non-chordtones proved its effectiveness in recognizing chord type patterns. Amortization with the original model, as well as experiments that incorporated the Truncated Dirichlet Process (TDP) and a two-stage training process, were conducted after the success of the simplified model. The use of the Truncated Dirichlet Process allowed the model to determine the optimal number of clusters that represents chord types autonomously. This method adjusted the number of clusters to fit the complexities of musical data. The two-stage training approach first focused on unbiased exploration and then on data-based adjustments. Specifically, the first stage sampled chord types from a uniform distribution to avoid bias, and the second stage fine-tuned the model based on observed data. The experiments have demonstrated that amortized inference is successful in recognizing unlabeled musical data. The model has learned to recognize ambiguous chord types, understand their shared characteristics, and identify the ornaments of different chords. The experiments show that amortized Bayesian inference improves the ability to model how listeners perceive and categorize musical harmony from unlabeled examples. These insights provide a foundation for improving AI systems and cognitive models in music cognition. Future work includes developing a fully unsupervised model. This involves reducing more or even all the assumptions, such as the initial definition of chordtones and ornaments.
