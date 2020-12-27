# Welcome

이 강의에서 다룰 것은 총 5가지이다.

- supervised learning
- learning theory
- deep learning
- unsupervised learning
- reinforcement learning

각각에 대해 간단하게 설명을 해보자.

## Supervised learning

보통 Dataset을 주고서, 나중에 주어지는 input값에 대한 output을 예측하게 학습시키는 것을 보고,  supervised learning이라 한다. house price problem이나 tumer classification problem 등이 이에 해당한다.

그리고 이는 **Regression Problem**과 **Classification Problem** 크게 두 갈래로 나눠지게 된다.

- Regression Problem의 경우에는 특정 input에 대한 output으로 value가 나오는 것이고,
    - 이에 대해서 linear Regression, ...에 대해서 CS229에서 다룰 것이다.
- Classification Problem의 경우에는 특정 input에 대한 output으로 type이 나오는 것이다. (굳이 type이 아니더라도, true/false 등 분류값, 대표값으로 나눠지는 것)

또 특징으로, Dataset이 주어졌을 때, 그 경향성이나 값 예측 하는 것이 보통 이에 해당되는 데, 이때 input이 n-dimension으로 주어지게 된다. 

머신러닝을 할 때, input값이 1~2 차원으로 주어지는 것은 오히려 그 경우가 적다. 무한-차원으로 주어질 수도 있는 데, 이런 것들을 처리하는 알고리즘을 보고 Support Vector Machine 이라 한다.

자율 주행에 대해서 훈련시키는 것도 이에 해당한다. 여러 (input, output)을 주어주고서, 이에 대해서 다음 input값에 대해 정확한 output을 예측하는 것이기 때문. 물론 이경우에는 input에 아주 많은 여러가지 vector들이 들어가게 될 것이다.

하지만, 결국 이 방법의 핵심은 다음과 같다. 훈련 중에 (input, output) Dataset을 많이 줘서 훈련을 시킨 후 Final Mapping을 성공시키는 것이다. 

## Learning Theory

머신러닝 전략에 대해서 다룰 것이다.

머신러닝을 하다보면, 그 결과에 영향을 미치는 인자가 굉장히 많다. 

Dataset을 수집하는 과정에서 편향되지는 않았는 지, Dataset의 양은 방대한지, Dataset에 오류는 없는 지, GPU의 속도는 빠른 지, 훈련시간은 충분히 주었는 지, 사용된 툴이 고려되는 환경에 적합한지, ... 

등 그 결과에 영향을 미치는 인자들이 굉장히 많은 데, 이에 대해서 어떻게 처리를 하고 학습을 시켜야 하는 지 배울 것

## Deep learning

딥러닝에 대해서는 230에서 더 자세히 다룰 것이다. (말고 설명에 대한 것은 없었음 ㅇㅅㅇ...)

## Unsupervised Learning

supervised learning과의 차이점은 output을 제대로 주지 않는 다는 것. 또, input값이 구별되어 있지 않는 다는 것이다.

예를 들어 소셜집단에서 특징을 찾아내는 것, 마켓에서 소비자의 경향성을 찾아내는 것, 뉴스에서 같은 뉴스끼리 묶는 것, 여러개의 음성이 합쳐진 곳에서 각각의 음성을 분리하는 것(Independent Component Analysis), ... 등이 모두 unsupervised Learning에 속한다.

이는 supervised와 다르게, 새로운 값에 대해서 예측을 하는 것이 아니라, Dataset이 주어졌을 때 이를 분류하는 것이다. 대신 Dataset에 대해서 input이 자세하게 주어지는 것이 아니라 그냥 투박하게 주어지고, output에 대해서도 어떤 분류되어 있는 값이 아님.

그냥 적당히 매핑되어있는 값들에 대해서 분류를 하는 것을 보고 unsupervised Learning이라 한다.

## Reinforcement Learning

예를 들어,

우리는 강아지가 어떻게 걸어야 optimize하게 걷는 것인지 잘 모른다. 어떻게 해야 강아지의 몸에 맞게 행동을 하는 것인지 모른다. 그래서 이 부분은 강아지에게 맡겨두는 걸로 한다. 단지 우리는 강아지가 좋은 행동을 하면 칭찬을 해주고, 나쁜 행동을 하면 화를 내는 것이다.

그렇게 되면, 강아지는 알아서 칭찬을 받을 만한 행위를 optimize하게 한다. 즉, 그 과정에 대해서는 우리가 잘 모르니 스스로 결정하게 하고 결과값만 우리가 원하는 방향으로 이끄는 것이다.

이런 학습방법을 보고 Reinforcement learning이라 한다.

모든 중간 과정에 대해서 우리가 핸들링을 할 수는 없으니, 결과값만 우리가 원하는 방향으로 좋고 나쁨을 알려줘서 좋은 결과값이 나오도록 하되 그 중간 과정은 관여하지 않는 것이다. 이 경우에 중간과정은 알아서 optimize한 행위를 하게 될 것이다.