## Sentiment of 3,000 SentiMix test tweets

| System | Hinglish labels used | Accuracy | Macro F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|---|
| Always "neutral" | no | 36.7 | 17.9 | 0.0 | 53.7 | 0.0 |
| VADER on the Hinglish tweet | no | 47.5 | 45.1 | 31.3 | 42.3 | 61.6 |
| VADER on our translation (run_f) | no | 49.2 | 46.7 | 40.7 | 37.8 | 61.7 |
| Classifier trained on labelled Hinglish tweets | yes, 14,000 | 67.6 | 68.1 | 70.7 | 61.1 | 72.6 |

## Paired bootstrap, translated minus untranslated, 1000 samples

| Metric | Difference | 95% interval | p-value |
|---|---|---|---|
| Macro F1 | +1.6 | -0.3 to +3.5 | 0.049 |
| Accuracy | +1.7 | -0.1 to +3.4 | 0.032 |

## By how mixed the tweet is (VADER untranslated, translated, difference)

| Group | Tweets | Macro F1 untranslated | Macro F1 translated | Difference | Accuracy untranslated | Accuracy translated | Difference |
|---|---|---|---|---|---|---|---|
| Hindi only | 173 | 19.9 | 44.7 | +24.8 | 33.5 | 45.7 | +12.1 |
| Mixed, low (CMI up to 20) | 1331 | 41.8 | 45.2 | +3.4 | 43.7 | 47.4 | +3.8 |
| Mixed, high (CMI above 20) | 1325 | 49.7 | 47.9 | -1.8 | 52.1 | 50.7 | -1.4 |
| English only | 171 | 42.0 | 39.9 | -2.1 | 56.1 | 55.0 | -1.2 |
| All | 3000 | 45.1 | 46.7 | +1.6 | 47.5 | 49.2 | +1.7 |

## VADER on the Hinglish tweet: true label (rows) against given label (columns)

| | negative | neutral | positive |
|---|---|---|---|
| negative | 209 | 489 | 202 |
| neutral | 172 | 484 | 444 |
| positive | 54 | 214 | 732 |

## VADER on our translation: true label (rows) against given label (columns)

| | negative | neutral | positive |
|---|---|---|---|
| negative | 309 | 250 | 341 |
| neutral | 245 | 339 | 516 |
| positive | 66 | 106 | 828 |

## Examples where translation corrected the label

| Tweet | Translation | True | Untranslated | Translated |
|---|---|---|---|---|
| @454dkhan @Heisunberg _ Agr kse ko itni importantce chaeay ni tou 😅 | @454dkhan @Heisunberg _ if you don't get so much importantce | neutral | positive | neutral |
| RT @DHEERAJ48968067 @nirahua1 @msunilbishnoi Wah bhai Shandaar jabab | retweet @DHEERAJ48968067 @nirahua1 @msunilbishnoi wow brother shan shan jabab | positive | neutral | positive |
| Modi ka lehar kisano pe baras raha kehar tut rahe haath pao road ki buri halato ka aasar @aajtak @BBCBreaking … https://t.co/link | modi's work is roaming on farmer and handle a lot of troubles on the road @aajtak @BBCBreaking ... https://t.co/link | negative | neutral | negative |
| RT @ImUrsTarun Zinda Hoon Main Tujhme Tujhme Rahunga Zinda Todke Sab Zanjeerein Main Aazad Parinda 🔥 One Of The Best Song Ever 💥 🔥 GO … | retweet @ImUrsTarun i am alive in you, you will see you in the end of my heart, i am feeling happy. one of the best song ever go go | positive | negative | positive |
| @akki _ dhoni @_ iRajeev You are stalking me bro . Just shut up and nikal . Le . Faltu logo ki mention mein jagah nahi haii | @akki _ dhoni @_ irajeev you are stalking me bro . just shut up and nikal . take . there is no place in the mention of false people | negative | neutral | negative |
| @narendramodi government job ke bare me sharkar ku responsible nahi vacancy itni kam aur vo final nahi kar va pa r … https://t.co/link | @narendramodi the government is not responsible for the job of vacance, but not so much and he won't final the final. https://t.co/link | negative | positive | negative |
| Bhai ye to sahi h . mgr hamare pas h hi kya . Bjp ne to achut bna hi diya h . Nfrat bhir hi di logon ke dilon me . Khuda … https://t.co/link | brother this is true. what is ours. bjp has made us good. in the hearts of people of god, god. https://t.co/link | positive | neutral | positive |
| @HamidMirPAK Sir leave ..... maza nae ha Pakistan ma kici b media journalist ka .. q k yahn koe such Bolta e nae | @HamidMirPAK sir leave ..... it is not fun pakistan. ki ki b media journalist ka .. q ya ya ko ko such suchta | negative | positive | negative |
| @Zorawar68324101 @danvir _ chauhan Hindi Rajbhasha he Rashtrabhasa nahi Kripya 10wi tak ki padhai kare | @Zorawar68324101 @danvir _ chauhan is not a hindi rajbhasha, please don't study till 10wi | positive | neutral | positive |
| EID ke iss Mubarak mauke par Unik Bazar lekar aaya hai aapke liye dhamekedaar offers Karein 3000 Rs ki khareedari … https://t.co/link | this time of eid has come to see aikikzar on you. forgive offers for buying 3000 rs. https://t.co/link | positive | neutral | positive |

## Examples where translation made the label wrong

| Tweet | Translation | True | Untranslated | Translated |
|---|---|---|---|---|
| Samarthan ke aabhav aur rajnitik swartho ke karan iska kabhi bhi sahi nyay nahi Kiya Gaya roshavash dharmik imarat … https://t.co/link | he was supporting support of the support and the slippers did not have done a good job roamvashing. https://t.co/link | neutral | neutral | positive |
| @QasimalirazaAli Ye such bolne walo ko mar sakte lekin such ko daba nhi sakhte or such ko jitna dabane ki koshish k … https://t.co/link | @QasimalirazaAli they can die these people, but can not be able to press such and try to press such things. https://t.co/link | neutral | neutral | negative |
| JcdPs moulladad k hudood Goth faqeerabad me 3 chor chori karty hue Gaon walon k hath pakry gaye 1 Farar police ky hawaly . Nabi bux Pandrani | jcdps mou hadood goth in moulladad, 3 minutes in faqeerabad went to pakistani's hand. it was a police police. nabi bux pandrani | neutral | neutral | positive |
| @priyankagandhi Esa koi exam nahi up govt . M jo in 2 saal m leak na huaa hu Ye to sarkar ki rajniti h ki na to ham … https://t.co/link | @priyankagandhi he does not have any exam that i am not able to take a 2 years in this 2 years, it is the politics of the government. https://t.co/link | negative | negative | neutral |
| Ab btaao ki isko samprdaayik roop kisne diya us ldke ne ya muslmaan ne ? @KapilMishra _ IND @sardanarohit … https://t.co/link | now what ldke has made it a property of love and muslmaan? @KapilMishra | neutral | neutral | positive |
| @mahisub Gud mrng Actually aapka msg box show naHi kar raha hai otherwise mai aapko msg hi karta Aap mjhe ek msg … https://t.co/link | @mahisub good mrng actually, you are not show your msg box otherwise otherwise i will give you msg and you will give me a msg ... https://t.co/link | neutral | neutral | positive |
| @its _ fatiiiii Nhi yaar yahan tw 1year walo k bhe khatam ho gye thy | @its _ there is no problem, the news of twit people here has gone over. | neutral | neutral | positive |
| @RubikaLiyaquat Ek dum sahi kaha TV news ke jariye naam kaam liya paisa bana liya or ab boycott ka natak ... Joker saale | @RubikaLiyaquat one is right, where did you make the name in the name of tc news, and now boycott's drama ... joker saale | negative | negative | positive |
| @IamSRKfanbig @RahulVerma4860 @iamsrk @Atlee _ dir Pehle chalta tha ab nahi pichle 2 saal ki Bollywood ki track check … https://t.co/link | @IamSRKfanbig @RahulVerma4860 @iamsrk @Atlee _ it was very good, now he would have a track of bollywood last year. https://t.co/link | neutral | neutral | positive |
| RT @bhumika7677 ghutan si hone lagi uss ke paas jate hue main khud se rooth gaya hun usey manaate hue ~ Azhar Iqbal | retweet @bhumika7677 i have gone to the neighbours that i have lost, i am afraid of saying that ~ az i ibal | neutral | neutral | negative |
