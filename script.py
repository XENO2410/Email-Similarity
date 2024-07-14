from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Fetch the training data for the specified categories
train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], 
                                  subset='train', shuffle=True, random_state=108)

# Print the content of the 6th email in the training set
print(train_emails.data[5])

# Print the target label of the 6th email in the training set
print(train_emails.target[5])

# Print the names of the target categories
print(train_emails.target_names)

# Fetch the test data for the same categories
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], 
                                 subset='test', shuffle=True, random_state=108)

# Initialize the CountVectorizer
counter = CountVectorizer()

# Fit the vectorizer to the combined data of both training and test sets
counter.fit(test_emails.data + train_emails.data)

# Transform the training data into count vectors
train_counts = counter.transform(train_emails.data)

# Transform the test data into count vectors
test_counts = counter.transform(test_emails.data)

# Initialize the Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier with the training data
classifier.fit(train_counts, train_emails.target)

# Evaluate the classifier on the test data and print the accuracy score
score = classifier.score(test_counts, test_emails.target)
print(score)
