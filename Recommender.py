from __future__ import division
import itertools
import pickle
import datetime
import hashlib
import locale
import numpy as np
import pycountry
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd

from collections import defaultdict
from sklearn.preprocessing import normalize

class DataCleaner:
    'Common utilities for converting strings to equivalent numbers or number buckets.'

    def __init__(self):
        #Loading 1ocales
        self.localeIdMap = defaultdict(int)
        for i, l in enumerate(locale.locale_alias.keys()):
            self.localeIdMap[l] = i + 1
        #Loading countries
        self.countryIdMap = defaultdict(int)
        ctryIdx = defaultdict(int)
        for i, c in enumerate(pycountry.countries):
            self.countryIdMap[c.name.lower()] = i + 1
            if c.name.lower() == "usa":
                ctryIdx["US"] = i
            if c.name.lower() == "canada":
                ctryIdx["CA"] = i
        for cc in ctryIdx.keys():
            for s in pycountry.subdivisions.get(country_code=cc):
                self.countryIdMap[s.name.lower()] = ctryIdx[cc] + 1
        # Loading gender id dictionary
        self.genderIdMap = defaultdict(int, {"male": 1, "female": 2})

    def getLocaleId(self, locstr):
        return self.localeIdMap[locstr.lower()]

    def getGenderId(self, genderStr):
        return self.genderIdMap[genderStr]

    def getJoinedYearMonth(self, dateString):
        dttm = datetime.datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%S.%fZ")
        return "". join([str(dttm.year), str(dttm.month)])

    def getCountryId(self, location):
        if (isinstance(location, str)
                and len(location.strip()) > 0
                and location.rfind(" ") > -1):
            return self.countryIdMap[location[location.rindex(" ") + 2:].lower()]
        else:
            return 0

    def getBirthYearInt(self, birthYear):
        try:
            return 0 if birthYear == "None" else int(birthYear)
        except:
            return 0

    def getTimezoneInt(self, timezone):
        try:
            return int(timezone)
        except:
            return 0

    def getFeatureHash(self, value):
        if len(value.strip()) == 0:
            return -1
        else:
            # Encode the Unicode string to bytes using UTF-8 encoding
            value_bytes = value.encode('utf-8')
            # Hash the bytes
            hash_value = hashlib.sha224(value_bytes).hexdigest()[0:4]
            # Convert the hexadecimal hash to an integer
            return int(hash_value, 16)

    def getFloatValue(self, value):
        if len(value.strip()) == 0:
            return 0.0
        else:
            return float(value)

#2. Handling user and event-related data

class ProgramEntities:
    # We only care the user and event appeared in train and test
    def __init__(self):
        # Count how many unique users' events are in the training dataset
        uniqueUsers = set()
        uniqueEvents = set()
        eventsForUser = defaultdict(set)
        usersForEvent = defaultdict(set)
        for filename in ["train.csv", "test.csv"]:
            f = open(filename,'r')
            f.readline().strip().split(",")
            for line in f:
                cols = line.strip().split(",")
                uniqueUsers.add(cols[0])
                uniqueEvents.add(cols[1])
                eventsForUser[cols[0]].add(cols[1])
                usersForEvent[cols[1]].add(cols[0])
            f.close()
        self.userEventScores = ss.dok_matrix((len(uniqueUsers), len(uniqueEvents)))
        self.userIndex = dict()
        self.eventIndex = dict()
        for i, u in enumerate(uniqueUsers):
            self.userIndex[u] = i
        for i, e in enumerate(uniqueEvents):
            self.eventIndex[e] = i
        ftrain = open("train.csv", 'r')
        ftrain.readline()
        for line in ftrain:
            cols = line.strip().split(",")
            i = self.userIndex[cols[0]]
            j = self.eventIndex[cols[1]]
            self.userEventScores[i, j] = int(cols[4]) - int(cols[5])
        ftrain.close()
        sio.mmwrite("PE_userEventScores", self.userEventScores)
        # To prevent unnecessary computation, we identify all associated users or associated events.
        # Associated users refer to
        # user pairs that have taken action on the same event at least once.
        # Associated events refer to
        # event pairs that have been interacted with by the same user at least once.
        self.uniqueUserPairs = set()
        self.uniqueEventPairs = set()
        for event in uniqueEvents:
            users = usersForEvent[event]
            if len(users) > 2:
                self.uniqueUserPairs.update(itertools.combinations(users, 2))
        for user in uniqueUsers:
            events = eventsForUser[user]
            if len(events) > 2:
                self.uniqueEventPairs.update(itertools.combinations(events, 2))
        pickle.dump(self.userIndex, open("PE_userIndex.pkl", 'wb'))
        pickle.dump(self.eventIndex, open("PE_eventIndex.pkl", 'wb'))


# 3. User-to-User Similarity Matrix
class Users:
    '''Constructing User/User Similarity Matrix'''
    def __init__(self, programEntities, sim=ssd.correlation):
        cleaner = DataCleaner()
        nusers = len(programEntities.userIndex.keys())
        fin = open ("users.csv", 'r')
        colnames = fin.readline().strip().split(",")
        self.userMatrix = ss.dok_matrix((nusers, len(colnames) - 1))
        for line in fin:
            cols = line.strip().split(",")
            #Consider only the users appearing in train.csv.
            if cols[0] in programEntities.userIndex:
                i = programEntities.userIndex[cols[0]]
                self.userMatrix[i, 0] = cleaner.getLocaleId(cols[1])
                self.userMatrix[i, 1] = cleaner.getBirthYearInt(cols[2])
                self.userMatrix[i, 2] = cleaner.getGenderId(cols[3])
                self.userMatrix[i, 3] = cleaner.getJoinedYearMonth(cols[4])
                self.userMatrix[i, 4] = cleaner.getCountryId(cols[5])
                self.userMatrix[i, 5] = cleaner.getTimezoneInt(cols[6])
        fin.close()
        # Normalize the user matrix
        self.userMatrix = normalize(self.userMatrix, norm="l1", axis = 0, copy = False)
        sio.mmwrite("US_userMatrix", self.userMatrix)
        # Calculate the user similarity matrix, which will be used later
        self.userSimMatrix = ss.dok_matrix((nusers, nusers))
        for i in range(0, nusers):
            self.userSimMatrix[i, i] = 1.0
        for u1, u2 in programEntities.uniqueUserPairs:
            i = programEntities.userIndex[u1]
            j = programEntities.userIndex[u2]
            if (i, j) not in self.userSimMatrix:
                # Flatten the 2-D matrices to 1-D arrays
                user_vector_i = self.userMatrix.getrow(i).todense().A1
                user_vector_j = self.userMatrix.getrow(j).todense().A1
                usim = sim(user_vector_i, user_vector_j)
                self.userSimMatrix[i, j] = usim
                self.userSimMatrix[j, i] = usim
        sio.mmwrite("US_userSimMatrix", self.userSimMatrix)
# 4. User Social Relationship Mining.
class UserFriends:
    '''Identifying a user's friends - the idea is very simple:
    1) If you have more friends, you may be outgoing and more likely to participate in various activities.
    2) If your friends are attending an event, you might also join in
    '''
    def __init__(self, programEntities):
        nusers = len(programEntities.userIndex.keys())
        self.numFriends = np.zeros((nusers))
        self.userFriends = ss.dok_matrix((nusers, nusers))
        fin = open("user_friends.csv", 'r')
        fin.readline() # skip header
        ln = 0
        for line in fin:
            if ln % 200 == 0:
                print("Loading line: ", ln)
            cols = line. strip().split(",")
            user = cols[0]
            if user in programEntities.userIndex:
                friends = cols[1].split(" ")
                i = programEntities.userIndex[user]
                self.numFriends[i] = len(friends)
                for friend in friends:
                    if friend in programEntities.userIndex:
                        j = programEntities.userIndex[friend]
                        # the objective of this score is to infer the degree to
                        # and direction in which this friend will influence the
                        # user's decision, so we sum the user/event score for
                        # this user across all training events.
                        eventsForUser = programEntities.userEventScores.getrow(j).todense()
                        score = eventsForUser.sum() / np.shape(eventsForUser)[1]
                        self.userFriends[i, j] += score
                        self.userFriends[j, i] += score
                        ln += 1
            ln += 1
        fin.close()
        #Normalize the array
        sumNumFriends = self.numFriends.sum(axis=0)
        self.numFriends = self.numFriends / sumNumFriends
        sio.mmwrite ("UF_numFriends", np.matrix(self.numFriends))
        self.userFriends = normalize(self.userFriends, norm="l1", axis=0, copy=False)
        sio.mmwrite ("UF_userFriends", self.userFriends)

# 5. Construct event-event similarity data
class Events:
    '''
    Construct event-event similarity, noting that there are two types of similarity here:
    1) Similarity calculated based on user-event behavior, similar to collaborative filtering.
    2) Similarity calculated based on the content of the events themselves (event information).
    '''
    def __init__(self, programEntities, psim=ssd.correlation, csim=ssd.cosine):
        cleaner = DataCleaner()
        fin=open("events.csv", 'r')
        fin.readline()  # skip header
        nevents = len(programEntities.eventIndex.keys())
        self.eventPropMatrix = ss.dok_matrix((nevents, 7))
        self.eventContMatrix = ss.dok_matrix((nevents, 100))
        ln = 0
        for line in fin.readlines():
            cols = line.strip().split(",")
            eventId = cols[0]
            if eventId in programEntities.eventIndex:
                i = programEntities.eventIndex[eventId]
                self.eventPropMatrix[i, 0] = cleaner.getJoinedYearMonth(cols[2])  # start_time
                self.eventPropMatrix[i, 1] = cleaner.getFeatureHash(cols[3])  # city
                self.eventPropMatrix[i, 2] = cleaner.getFeatureHash(cols[4])  # state
                self.eventPropMatrix[i, 3] = cleaner.getFeatureHash(cols[5])  # zip
                self.eventPropMatrix[i, 4] = cleaner.getFeatureHash(cols[6])  # country
                self.eventPropMatrix[i, 5] = cleaner.getFloatValue(cols[7])  # lat
                self.eventPropMatrix[i, 6] = cleaner.getFloatValue(cols[8])  # lon
                for j in range(9, 109):
                    self.eventContMatrix[i, j - 9] = cols[j]
                ln += 1
        fin.close()
        self.eventPropMatrix = normalize(self.eventPropMatrix, norm = "l1", axis = 0, copy = False)
        sio.mmwrite("EV_eventPropMatrix", self.eventPropMatrix)
        self.eventContMatrix = normalize(self.eventContMatrix, norm = "l1", axis = 0, copy=False)
        sio.mmwrite("EV_eventContMatrix", self.eventContMatrix)
        #calculate similarity between event pairs based on the two matrices
        self.eventPropSim = ss.dok_matrix((nevents, nevents))
        self.eventContSim = ss.dok_matrix((nevents, nevents))
        for e1, e2 in programEntities.uniqueEventPairs:
            i = programEntities.eventIndex[e1]
            j = programEntities.eventIndex[e2]

            if (i, j) not in self.eventPropSim:
                # Flatten the 2-D matrices to 1-D arrays
                event_vector_i = self.eventPropMatrix.getrow(i).todense().A1
                event_vector_j = self.eventPropMatrix.getrow(j).todense().A1
                epsim = psim(event_vector_i, event_vector_j)
                self.eventPropSim[i, j] = epsim
                self.eventPropSim[j, i] = epsim

            if (i, j) not in self.eventContSim:
                # Flatten the 2-D matrices to 1-D arrays
                event_vector_i = self.eventContMatrix.getrow(i).todense().A1
                event_vector_j = self.eventContMatrix.getrow(j).todense().A1
                ecsim = csim(event_vector_i, event_vector_j)
                self.eventContSim[i, j] = ecsim  # Corrected from epsim
                self.eventContSim[j, i] = ecsim  # Corrected from epsim

        sio.mmwrite("EV_eventPropSim", self.eventPropSim)
        sio.mmwrite("EV_eventContSim", self.eventContSim)

# 6. Activity/Event Popularity Data
class EventAttendees():
    '''
    Count the number of people who attend and do not attend a specific event
    in preparation for assessing the event's activity level.
    '''
    def __init__(self, programEvents):
        nevents = len(programEvents.eventIndex.keys())
        self.eventPopularity = ss.dok_matrix((nevents, 1))
        f = open("event_attendees.csv", 'r')
        f.readline()  # skip header
        for line in f:
            cols = line.strip().split(",")
            eventId = cols[0]
            if eventId in programEvents.eventIndex:
                i = programEvents.eventIndex[eventId]
                self.eventPopularity[i, 0] = \
                    len(cols[1].split(" ")) - len(cols[4].split(" "))
        f.close()
        self.eventPopularity = normalize(self.eventPopularity, norm="l1", axis=0, copy=False)
        sio.mmwrite("EA_eventPopularity", self.eventPopularity)

# 7. Get together all data processing and preparation steps
def data_prepare():
    "Calculate and generate all the data, store it in a matrix or other format for easy feature extraction and modeling in subsequent steps."
    print("第1步：统计user和event相关信息⋯")
    pe = ProgramEntities()
    print("第1步完成...\n")

    print("第2步：计算用户相似度信息，并用矩阵形式存储.")
    Users(pe)
    print("第2步完成...\n")

    print("第3步：计算用户社交关系信息，并存储⋯")
    UserFriends(pe)
    print("第3步完成..\n")

    print("第4步：计算event相似度信息，并用矩阵形式存储")
    Events(pe)
    print("第4步完成..\n")

    print("第5步：计算event热度信息⋯.")
    EventAttendees(pe)
    print("第5步完成..1n^")

#Run the data preparation process.
data_prepare()

# 8. Feature Engineering
# This is the feature engineering section.
# from __future__ import division

import pickle
import numpy as np  # Import NumPy module
import scipy.io as sio

class DataRewriter:
    def __init__(self):
        #Read in the data for initialization
        self.userIndex = pickle.load(open("PE_userIndex.pkl", 'rb'))
        self.eventIndex = pickle.load(open("PE_eventIndex.pkl", 'rb'))
        self.userEventScores = sio.mmread("PE_userEventScores")
        self.userSimMatrix = sio.mmread("US_userSimMatrix")
        self.eventPropSim = sio.mmread("EV_eventPropSim")
        self.eventContSim = sio.mmread("EV_eventContSim")
        self.numFriends = sio.mmread("UF_numFriends")
        self.userFriends = sio.mmread("UF_userFriends")
        self.eventPopularity = sio.mmread("EA_eventPopularity")

    def userReco(self, userId, eventId):
        """Based on User-based collaborative filtering, the basic pseudocode idea to obtain event recommendations is as follows:
        for item i
            for every other user v that has a preference for i
                compute similarity s between u and v
                incorporate y's preference for i weighted by s into running average
        return top items ranked by weighted average"""
        i = self.userIndex[userId]
        j = self.eventIndex[eventId]

        # Convert the coo_matrix to a dense NumPy array
        userEventScores_dense = self.userEventScores.toarray()

        vs = userEventScores_dense[:, j]  # Access a column as a NumPy array
        sims = self.userSimMatrix.getrow(i).toarray()[0]  # Access a row as a NumPy array

        prod = sum(sims * vs)  # Calculate the sum of the element-wise product
        try:
            return prod - userEventScores_dense[i, j]
        except IndexError:
            return 0

    def eventReco(self, userId, eventId):
        """Based on Item-based collaborative filtering, the basic pseudocode idea to obtain event recommendations is as follows:
        for item i
            for every item j tht u has a preference for
                compute similarity s between i and j
                add u's preference for j weighted by s to a running average
        return top items, ranked by weighted average"""
        i = self.userIndex[userId]
        j = self.eventIndex[eventId]
        # Convert the coo_matrix to a dense NumPy array
        userEventScores_dense = self.userEventScores.toarray()

        js = userEventScores_dense[i, :]

        # Convert the coo_matrix to a dense NumPy array
        eventPropSim_dense = self.eventPropSim.toarray()
        eventContSim_dense = self.eventContSim.toarray()

        psim = eventPropSim_dense[:, j]  # Access a column as a NumPy array
        csim = eventContSim_dense[:, j]  # Access a column as a NumPy array

        pprod = js * psim
        cprod = js * csim
        pscore = 0
        cscore = 0
        try:
            pscore = pprod - userEventScores_dense[i, j]
        except IndexError:
            pass
        return pscore, cscore

    def userPop(self, userId):
        """Infer users' social activity based on the number of their friends,
        primarily considering that if a user has many friends,
        they may be more inclined to participate in various social activities."""
        if userId in self.userIndex:
            i = self.userIndex[userId]
            try:
                return self.numFriends[i]
            except IndexError:
                return 0
        else:
            return 0

    def friendInfluence(self, userId):
        """The influence of friends on users mainly considers how many of the user's friends are very enthusiastic about participating
        in various social activities or events.
        If a user's social circle consists of friends who actively participate in various events,
        it may have a certain impact on the current user."""
        nusers = np.shape(self.userFriends)[1]
        i = self.userIndex[userId]

        # Convert the COO matrix to a dense NumPy array
        userFriends_dense = self.userFriends.toarray()

        return (userFriends_dense[i, :].sum() / nusers)

    def eventPop(self, eventId):
        """The popularity of the event itself is primarily determined by the number of participants"""
        i = self.eventIndex[eventId]
        eventPopularity_dense = self.eventPopularity.toarray()
        return eventPopularity_dense[i, 0]

    def rewriteData(self, start=1, train=True, header=True):
        """Combine the previous user-based collaborative filtering, item-based collaborative filtering,
        as well as various popularity and influence metrics as features to generate new training data for use
        in a classifier for classification"""
        fn = "train.csv" if train else "test.csv"
        fin = open(fn, 'r')
        fout = open("data_" + fn, 'w', encoding='utf-8')
        # write output header
        if header:
            ocolnames = ["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
            if train:
                ocolnames.append("interested")
                ocolnames.append("not_interested")
            fout.write(",". join(ocolnames) + "\n")
        ln = 0
        for line in fin:
            ln += 1
            if ln < start:
                continue
            cols = line.strip().split(",")
            userId = cols[0]
            eventId = cols[1]
            invited = cols[2]
            if ln % 500 == 0:
                print("%s:%d (userId, eventId)=(%s, %s)" % (fn, ln, userId, eventId))

            user_reco = self.userReco(userId, eventId)
            evt_p_reco, evt_c_reco = self.eventReco(userId, eventId)
            user_pop = self.userPop(userId)
            frnd_infl = self.friendInfluence(userId)
            evt_pop = self.eventPop(eventId)
            ocols = [invited, user_reco, evt_p_reco, evt_c_reco, user_pop, frnd_infl, evt_pop]
            if train:
                ocols.append(cols[4])  # interested
                ocols.append(cols[5])  # not_interested
            fout.write(",".join(map(lambda x: str(x), ocols)) + "n")
        fin.close()
        fout.close()

    def rewriteTrainingSet(self):
        self.rewriteData(True)

    def rewriteTestSet(self):
        self.rewriteData(False)

# When running with python, the actual class will be converted to a . so
# file, and the following code (along with the commented out import below)
# will need to be put into another py and this should be run.

# import CRegressionData as rd
dr = DataRewriter()
print("Generate training data...ln")
dr.rewriteData(train=True, start=2, header=True)
print("Generate prediction data...ln")
dr.rewriteData(train=False, start=2, header=True)

# Step 9
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier

def train():
    trainDf = pd.read_csv("data_train.csv", sep=',')
    X = np.array(trainDf[["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]])
    y = np.array(trainDf["interested"])
    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(X, y)
    return clf

def validate():
    trainDf = pd.read_csv("data_train.csv")
    X = np.array(trainDf[["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]])
    y = np.array(trainDf["interested"])
    nrows = len(trainDf)
    kfold = KFold(n_splits=10)
    avgAccuracy = 0
    run = 0
    for train, test in kfold.split(X):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf = SGDClassifier(loss="log", penalty="l2")
        clf.fit(Xtrain, ytrain)
        accuracy = clf.score(Xtest, ytest)
        print("accuracy (run %d): %f" % (run, accuracy))
        avgAccuracy += accuracy
        run += 1
    print("Average accuracy", (avgAccuracy / run))

def test(clf):
    origTestDf = pd.read_csv("test.csv")
    users = origTestDf.user
    events = origTestDf.event
    testDf = pd.read_csv("data_test.csv")
    fout = open("result.csv", "w")  # Use 'w' for write mode
    fout.write(",".join(["user", "event", "outcome", "dist"]) + "\n")
    nrows = len(testDf)
    Xp = np.array(testDf)
    yp = np.zeros((nrows, 2))
    for i in range(0, nrows):
        xp = Xp[i, :]
        yp[i, 0] = clf.predict(xp.reshape(1, -1))  # Reshape to ensure a 2D array
        yp[i, 1] = clf.decision_function(xp.reshape(1, -1))
        fout.write(",".join(map(lambda x: str(x), [users[i], events[i], yp[i, 0], yp[i, 1]])) + "\n")
    fout.close()

clf = train()
validate()  # Call the validate function to perform cross-validation
test(clf)  # Call the test function to generate predictions for the test data

# 10.Generate the file(s) to be submitted
# from __future__ import division

import pandas as pd

def byDist(x, y):
    return int(y[1] - x[1])

def generate_submission_file():
    # Output file
    fout = open('final_result.csv', 'w')  # Use 'w' for write mode
    fout.write(",".join(["User", "Events"]) + "\n")  # Use '\n' for a newline
    resultDf = pd.read_csv("result.csv")

    # Group remaining user/events
    grouped = resultDf.groupby("user")

    for name, group in grouped:
        user = str(name)
        tuples = zip(list(group.event), list(group.dist), list(group.outcome))
        tuples = sorted(tuples, key=lambda x: x[1])
        events = "\"" + ",".join(map(lambda x: x[0], tuples)) + "\""
        fout.write(",".join([user, events]) + "\n")

    fout.close()


# Call the function to generate the submission file
generate_submission_file()
