import os

family = "wikipedia"
mylang = "es"
usernames["wikipedia"]["es"] = "PCendrerBOT"
usernames["wikipedia"]["test"] = "PCendrerBOT"

authenticate["*.wikipedia.org"] = (
    os.environ["WIKIMEDIA_CONSUMER_KEY"],
    os.environ["WIKIMEDIA_CONSUMER_SECRET"],
    os.environ["WIKIMEDIA_ACCESS_KEY"],
    os.environ["WIKIMEDIA_ACCESS_SECRET"],
)
