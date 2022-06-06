# Fair-Performance-based-User-Recommendation-in-eCoaching-Systems

## Abstract

Offering timely support to users in eCoaching systems is a key factor to keep them engaged. However, coaches usually follow a lot of users, so it is hard for them to prioritize those with whom they should interact first. Timeliness is especially needed when health implications might be the consequence of a lack of support. In this paper, we focus on this last scenario, by considering an eCoaching platform for runners. Our goal is to provide a coach with a ranked list of users, according to the support they need.
Moreover, we want to guarantee a fair exposure in the ranking, to make sure that users of different groups have equal opportunities to get supported. In order to do so, we first model their performance and running behavior, and then present a ranking algorithm to recommend users to coaches, according to their performance in the last running session and the quality of the previous ones.
We provide measures of fairness that allow us to assess the exposure of users of different groups in the ranking, and propose a re-ranking algorithm to guarantee a fair exposure. Experiments on data coming from the previously mentioned platform for runners show the effectiveness of our approach on standard metrics for ranking quality assessment and its capability to provide a fair exposure to users.

## Installation

Install Python (>=3.6):

```
    sudo apt-get update
    sudo apt-get install python3.6
```

Clone this repository:

```$ git clone https://github.com/wiguider/Fair-Performance-based-User-Recommendation-in-eCoaching-Systems.git```

Install the requirements:

```$ pip install -r requirements.txt```
