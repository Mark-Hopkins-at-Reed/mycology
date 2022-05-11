from naivebayes import train_naive_bayes
import pandas as pd

FEATURES = {
     'cap-shape':                "bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s",
     'cap-surface':              "fibrous=f,grooves=g,scaly=y,smooth=s",
     'cap-color':                "brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y",
     'bruises':                  "bruises=t,no=f",
     'odor':                     "almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s",
     'gill-attachment':          "attached=a,descending=d,free=f,notched=n",
     'gill-spacing':             "close=c,crowded=w,distant=d",
     'gill-size':                "broad=b,narrow=n",
     'gill-color':               "black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y",
     'stalk-shape':              "enlarging=e,tapering=t",
     'stalk-root':               "bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?",
     'stalk-surface-above-ring': "fibrous=f,scaly=y,silky=k,smooth=s",
     'stalk-surface-below-ring': "fibrous=f,scaly=y,silky=k,smooth=s",
     'stalk-color-above-ring':   "brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y",
     'stalk-color-below-ring':   "brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y",
     'veil-type':                "partial=p,universal=u",
     'veil-color':               "brown=n,orange=o,white=w,yellow=y",
     'ring-number':              "none=n,one=o,two=t",
     'ring-type':                "cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z",
     'spore-print-color':        "black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y",
     'population':               "abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y",
     'habitat':                  "grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d"
}

def extract_feature_abbreviations():
    feature_abbreviations = dict()
    for feature in FEATURES:
        abbrev_strs = FEATURES[feature].split(',')
        abbrev_map = {k: v for (v, k) in [a.split('=') for a in abbrev_strs]}
        feature_abbreviations[feature] = abbrev_map
    return feature_abbreviations


def elicit_observations():
    def choose_feature():
        print('Please choose a feature that you have observed (or enter a blank line to continue):')
        for feature in sorted(feature_glossary):
            print(f'  {feature}')
        if len(observations) > 0:
            print('Observations so far:')
            for observed in observations:
                print(f'  {observed}: {feature_glossary[observed][observations[observed]]}')
        while True:
            observed = input('> ').strip()
            if observed  == "" or observed  in feature_glossary:
                return observed
            else:
                print(f'Unrecognized feature: {observed}. Try again.')

    def specify_observation():
        print('Specify your observation:')
        for feature_value in FEATURES[observed_feature].split(','):
            print(f'  {feature_value}')
        while True:
            observation = input('> ').strip()
            if observation in feature_glossary[observed_feature]:
                return observation
            else:
                print(f'Unrecognized value: {observation}. Try again.')

    feature_glossary = extract_feature_abbreviations()
    observations = dict()
    while True:
        observed_feature = choose_feature()
        if observed_feature == "":
            break
        observations[observed_feature] = specify_observation()
    return observations


def main():
    print("HELLO, I AM MUSHROOM SAFETY PROTOCOL P1LZ3. BOOTING UP...")
    data = pd.read_csv('mushrooms/agaricus-lepiota.csv')
    naive_bayes = train_naive_bayes(data, 'poisonous')
    while True:
        observations = elicit_observations()
        posterior = naive_bayes.posterior(observations)
        print(f'THERE IS A PROBABILITY OF {round(posterior["p"]*100, 2)}% THAT THIS MUSHROOM IS POISONOUS.')
        input("Press enter to continue.")

if __name__ == '__main__':
    main()
