
import pymongo
import datetime, pprint
from pymongo import MongoClient

client = pymongo.MongoClient("mongodb+srv://aorso:ilovecats@airbnb.kgupgez.mongodb.net/?retryWrites=true&w=majority")
db = client.sample_airbnb



#Getting a collection called posts
posts = db.listingsAndReviews

#print(posts)

collection_name = db.list_collection_names()


#Q1

top5 = posts.aggregate([
    {"$unwind":"$host"},
    {"$group": {
    "_id": ["$host.host_listings_count","$host.host_name"]}},{"$sort": {"_id": -1}},{"$limit":5}])

for entry in top5:
	pprint.pprint(entry)

#Q1 Alternative Method

hosting = posts.find({},{"host.host_name": 1,"host.host_listings_count": 1}).sort("host.host_listings_count",-1)

ls =[]
i=1 
for a in hosting:
	entry = a["host"]["host_name"],a["host"]["host_listings_count"]
	if i >  5:
		break
	if entry not in ls:
		ls.append(entry)
		i+=1
	else:
		continue

for l in ls:
	print(l)

#Q2

wifi_min = posts.find({"amenities": {"$all": ["Cable TV", "Iron", "Wifi"]}}, 
	{"_id": 1, "name": 1, "property_type": 1, "minimum_nights": 1}).sort("minimum_nights", -1).limit(10)

for wi_min in wifi_min:
	pprint.pprint(wi_min)

#Q3

wifi_max = posts.find({}, 
	{"_id": 1, "name": 1, "property_type": 1, "maximum_nights": 1}).sort("maximum_nights", -1).limit(1)

for wi_max in wifi_max:
	pprint.pprint(wi_max)

#Q4

accom = posts.find({"$and":[{"accommodates":{"$gte":5,"$lte":8}},{"beds":{"$gt": 4}}]},
	{"_id": 1, "name": 1, "property_type": 1, "accommodates":1, "beds": 1,"bedrooms":1}).sort("accommodates",-1).limit(5)


for a in accom:
	pprint.pprint(a)

#Q5

for post in posts.find( { "$expr":{"$gt":["$bathrooms","$bedrooms"]}},
							 {"_id":1, "name":1, "property_type":1, "bathrooms": 1, "bedrooms":1, "beds":1}).sort("bathrooms", -1).limit(5):
	  pprint.pprint(post)

#Q6

for top10 in posts.find({},{"_id":1, "name":1, "property_type":1, "summary":1, 
	"review_scores.review_scores_rating":1}).sort("review_scores.review_scores_rating",-1).limit(10):
        pprint.pprint(top10)


