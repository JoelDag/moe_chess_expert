# MoE-LPR Implementation plan 

MoE-LPR implementation plan based on this paper
https://arxiv.org/html/2408.11396v1

* 1. Upcycle the base model. They take the FFNs of the base model and copy paste if -> thus they create multiple experts where the original FFN becomes expert 0
* 2. They check on which languages the model performs bad (unseen langauges) and train the MoE upcycled model on these new languages, the router also trains but only on the new languages, since its upcycled it hasnt seen the original languages
* 3. Since the router is really biased to the new langauges, they now retrain on 1% of the original languages, the model has seen before, but they only train the router to refine the routing.
They use an indicator function to check if the languae was seen before or not, if it was seen before, they route in a way, the original data gets routed to expert 0 (usually / ideally). In this step they include original languages as well as newly added languages to achieve optimal routing
'