### Introduction:

Ever wanted to jump into the conversation at work about the latest _Law & Order_? Or met someone new that you that you really like and think could be _the one_...and they also just so happen to be really into _The Simpsons_?

If you’ve got a few hours, Warp10 has you covered: it’s a hybrid recommender designed to suggest a thematically diverse and representative sampling of a given TV series in a user-defined number of episodes. For this project, I used _Star Trek_ as my test case, given its fairly intimidating 719 episode run from The Original Series to Enterprise.


### So, what does it do?

Unlike typical recommenders, which return suggestions based on a user’s consumption patterns or similarity to other users, Warp10 champions variety, giving you the breadth and depth of an episodic series, and in turn, more conversational topics to deploy at work tomorrow or on that next date.

Designed in Python, Warp10 utilizes SKLearn and Surprise’s k-Means clustering and NMF algorithms to create “topic clusters” from IMDb user reviews, ratings, and episode summaries in two distinct ways:

1.	Clustering episodes based on the most common terms in IMDb user review and episode summary text,
2.	and, using matrix factorization to tease out latent features or “hidden themes” via IMDb user ratings.

Warp10 then returns a user-defined number of episodes from a random selection of both algorithms’ picks.


### Challenges:

Given _Star Trek’s_ largely episodic format, it’s easy to recommend episodes as most require very little context before viewing. This won’t work for television shows like _Game of Thrones_ or _Breaking Bad_ however, making serialized television Warp10’s biggest challenge.

There’s also the issue of judging Warp10’s recommendations as there seemed to be no perfect metric to assess recommendation variety and quality outside of human judgment. With the subjective nature of human taste however, tuning Warp10 to suit most people (let alone Trekkies) has not been a trivial task and will require further testing and perfecting. It’s not an insignificant endeavor, however, as we live doubly in the age of “Peak-TV” and the algorithmic sorting of our tastes into monocultural niches. Warp10 humbly aims to be a response.

Lastly, interpretability is also an issue, both with some of Warp10’s text clusters and the latent features it derives from matrix factorization. Like appraisal of its recommendations, Warp10 requires a fair bit of human judgment in determining what themes are singled out in its clusters, and yet more careful judgment in its otherwise inscrutable latent features.


### Future work:

Warp10 currently uses only two algorithms to split up a given series. I’m hoping to experiment with and include more in the future given that different algorithms cluster or derive latent features differently. Warp10 also needs to be generalized to be series agnostic so that it can be used with other television shows such as Law & Order, the Simpsons, South Park, etc. prior to it being released online. It’s my hope that once it’s out in the open, users can help direct Warp10’s development, both in the algorithms it employs and the tuning of its many parameters.
