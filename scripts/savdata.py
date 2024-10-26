import pandas as pd
from io import StringIO

data = """Tweet ID,User Handle,Tweet Text,Timestamp,Sentiment,Cleaned Text,Tokenized Text
123456,@user1,"I love this product!",2024-10-26 10:00:00,Positive,"love product","love;product"
123457,@user2,"This is terrible!",2024-10-26 10:01:00,Negative,"terrible","terrible"
123458,@user3,"It’s okay, not great.",2024-10-26 10:02:00,Neutral,"okay not great","okay;not;great"
123459,@user4,"Absolutely fantastic experience!",2024-10-26 10:03:00,Positive,"Absolutely fantastic experience","Absolutely;fantastic;experience"
123460,@user5,"Horrible service and rude staff.",2024-10-26 10:04:00,Negative,"Horrible service and rude staff","Horrible;service;rude;staff"
123461,@user6,"Not worth the price.",2024-10-26 10:05:00,Negative,"Not worth the price","Not;worth;price"
123462,@user7,"Average product, nothing special.",2024-10-26 10:06:00,Neutral,"Average product nothing special","Average;product;nothing;special"
123463,@user8,"Loved the design, very sleek!",2024-10-26 10:07:00,Positive,"Loved the design very sleek","Loved;the;design;very;sleek"
123464,@user9,"The worst experience I've had.",2024-10-26 10:08:00,Negative,"worst experience I've had","worst;experience;I've;had"
123465,@user10,"It's decent but has room for improvement.",2024-10-26 10:09:00,Neutral,"It's decent but has room for improvement","It's;decent;but;has;room;for;improvement"
"""

# Read the data into a DataFrame
df = pd.read_csv(StringIO(data))

# Save the DataFrame to a CSV file
df.to_csv('cleaned_data.csv', index=False)
