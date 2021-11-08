import json
import numpy as np
import tensorflow as tf
from itertools import combinations
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

class PlayerEncoder:
    def __init__(self):
        self._encodings = {}
        self.pcount_ = 0
        
    def fit(self, player_arrs):
        nextval = self.pcount_
        for arr in player_arrs:
            for player in arr:
                if player not in self._encodings.keys():
                    self._encodings[player] = nextval
                    nextval += 1
        self.pcount_ = nextval
                    
    def transform(self, player_arrs):
        result = np.zeros((len(player_arrs), self.pcount_), dtype=int)
        for i,arr in enumerate(player_arrs):
            indexes = [self._encodings[p] for p in arr]
            result[i, indexes] = 1
        return result
    
    def fit_transform(self, player_arrs):
        self.fit(player_arrs)
        return self.transform(player_arrs)
    
    def save(self, path):
        with open(path, 'w') as outfile:
            json.dump(self._encodings, outfile)
            
    @staticmethod
    def load(path):
        pe = PlayerEncoder()
        with open(path, 'r') as infile:
            pe._encodings = json.load(infile)
            pe.pcount_ = len(pe._encodings)
        return pe

class Predictor:
    __kill_distance = {
        'Short': 1,
        'Medium': 2,
        'Long': 3
    }

    __task_bar_updates = {
        'Always': 2,
        'Meetings': 1,
        'Never': 0
    }

    def __init__(self):
        # I'm basically treating these as any other custom parser, rather than actually
        # needing to fit an unknown data source
        self._menc = OneHotEncoder(sparse=False)
        self._wenc = OneHotEncoder(sparse=False)
        self._menc.fit([['The Skeld'], ['MIRA HQ'], ['Polus'], ['Airship']])
        self._wenc.fit([['crewmates'], ['impostors']])

    def fit(self, matches_with_results, *, epochs=100, stop_after=5):
        self._model = Sequential()
        self._penc = PlayerEncoder()
        self._penc.fit([m['crewmates'] + m['impostors'] for m in matches_with_results])

        inputs, targets = self.__preprocess(matches_with_results)
        pcount = self._penc.pcount_
        mcount = len(self._menc.categories_[0])
        self._model.add(Dense(pcount * pcount * mcount, input_shape=inputs[0].shape, activation='relu'))
        layer_size = pcount * pcount
        while layer_size > targets[0].shape[0]:
            self._model.add(Dense(layer_size, activation='relu'))
            layer_size //= 3
        
        self._model.add(Dense(targets[0].shape[0], activation='softmax'))
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        # Initial fit
        if stop_after > 0:
            stop = [EarlyStopping(monitor='val_acc', patience=5)]
        else:
            stop = None
        history = self._model.fit(
            x = inputs,
            y = targets,
            batch_size = 32,
            epochs = epochs,
            validation_split = 0.2,
            callbacks = stop
        )
        return history

    def fit_partial(self, matches_with_results, *, epochs=1):
        inputs, targets = self.__preprocess(matches_with_results)
        # One epoch, or two???
        return self._model.fit(
            x = inputs,
            y = targets,
            batch_size = 32,
            epochs = epochs
        )

    def get_leaders(self, settings, *, verbose=False):
        '''
        Gets the ranking of all known players for given game settings

        Settings must include:
            * map
            * player_count
            * impostor_count
            * confirm_ejects
            * emergency_meetings
            * emergency_cooldown
            * discussion_time
            * voting_time
            * anonymous_votes
            * player_speed
            * crewmate_vision
            * impostor_vision
            * kill_cooldown
            * kill_distance
            * visual_tasks
            * task_bar_updates
            * common_tasks
            * long_tasks
            * short_tasks
        '''
        # This might get expensive. Make sure the player counts are actually valid
        # before continuing
        if settings['player_count'] <= settings['impostor_count'] * 3:
            raise ValueError('Not enough players for impostor count')
        crew_count = settings['player_count'] - settings['impostor_count']

        # Use the known players to construct all possible matchups with these settings
        # I haven't actually done the counting for how many permutations that could
        # turn into. If it turns out there's just way too many possible matches then
        # I'll just take a subset or something.
        if verbose:
            print('Create impostor combinations')
        rng = np.random.default_rng()
        player_list = list(self._penc._encodings.keys())
        impostor_combs = list(combinations(player_list, settings['impostor_count']))
        impostors = rng.choice(impostor_combs, min(len(impostor_combs), 100), replace=False)
        matches = []
        match_inputs = []
        if verbose:
            print('Create crewmate combinations ', end='')
        iverb = 0
        lastverb = 0
        for imp in impostors:
            candidates = np.setdiff1d(player_list, imp)
            crewmate_combs = list(combinations(candidates, crew_count))
            crewmates = rng.choice(crewmate_combs, min(len(crewmate_combs), 1000), replace=False)
            limp = list(imp)
            for crew in crewmates:
                m = {
                    **settings,
                    'impostors': limp,
                    'crewmates': list(crew)
                }
                matches.append(m)
                match_inputs.append(self.__convert_match_pred(m))
            if verbose:
                iverb += 1
                if (iverb * 75 / len(impostors)) > lastverb:
                    print('*', end='')
                    lastverb += 1
        if verbose:
            print('\nGenerate results')
        results = self._model.predict(np.array(match_inputs))
        if verbose:
            print('Count wins')
        wins = {p:0 for p in player_list}
        for match,result in zip(matches, results):
            winner = self._wenc.inverse_transform([np.round(result)])[0,0]
            for p in match[winner]:
                wins[p] += 1
        return sorted(wins.items(), key=lambda x: x[1], reverse=True)

    def __preprocess(self, matches):
        inputs = []
        targets = []
        for m in matches:
            result = self.__convert_match_result(m)
            inputs.append(result[0])
            targets.append(result[1])
        inputs = np.array(inputs)
        targets = np.array(targets)
        return (inputs, targets)

    def __convert_match_pred(self, match):
        crewmates = list(self._penc.transform([match['crewmates']])[0])
        impostors = list(self._penc.transform([match['impostors']])[0])
        map_played = list(self._menc.transform([[match['map']]])[0])
        
        inputs = np.array(crewmates + impostors + map_played + [
            np.sum(crewmates), # No. crewmates
            np.sum(impostors), # No. impostors
            match['confirm_ejects'],
            match['emergency_meetings'],
            match['emergency_cooldown'],
            match['discussion_time'],
            match['voting_time'],
            match['anonymous_votes'],
            match['player_speed'],
            match['crewmate_vision'],
            match['impostor_vision'],
            match['kill_cooldown'],
            self.__kill_distance[match['kill_distance']],
            match['visual_tasks'],
            self.__task_bar_updates[match['task_bar_updates']],
            match['common_tasks'],
            match['long_tasks'],
            match['short_tasks']
        ])
        return inputs

    def __convert_match_result(self, match):
        inputs = self.__convert_match_pred(match)
        # Convert winners
        winner = self._wenc.transform([[match['match_winner']]])[0]
        
        return (inputs, winner)

    def summary(self):
        return self._model.summary()

    def save(self, path):
        # Write model
        self._model.save(f'{path}/model')
        self._penc.save(f'{path}/penc')
        # Don't bother saving menc or wenc.
        # They *should* be the same every time, right?

    @staticmethod
    def load(path):
        pred = Predictor()
        pred._model = tf.keras.models.load_model(f'{path}/model')
        pred._penc = PlayerEncoder.load(f'{path}/penc')
        return pred
