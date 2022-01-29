from experta import *

"""
Author: Justin Hu
Date: 2021/01/27
Python Version: 3.9
"""

"""
expert system 1
"""


class Cover(Fact):
    """Cover types: fur or feathers or None"""
    pass


class Wings(Fact):
    """Wings status: True or False"""


class AnimalChecker(KnowledgeEngine):
    """
    determine whether an animal is a bird, mammal, or unknown
    The results follow the rule below:
    Input:   Cover       Wings     ->   Output
            fur          True      ->   mammal
            fur         False      ->   Unspecified
            feathers    True        ->  bird
            feathers    False       ->  unknown
    """
    @Rule(Cover('feathers'), Wings(True))
    def is_bird(self):
        print('bird')

    @Rule(Cover('fur'), Wings(True))
    def is_mammal(self):
        print('mammal')

    @Rule(Cover('feathers'), Wings(False))
    def is_unknown(self):
        print('unknown')

    @Rule(Cover('fur'), Wings(False))
    def is_unspecified(self):
        print('unspecified')


"""
expert system 2
"""


class ProduceMilk(Fact):
    """ProduceMilk: True or False"""
    pass


class SingleCelled(Fact):
    """SingleCelled: True of False"""
    pass


class Vertebrate(Fact):
    """Vertebrate: True or False"""
    pass


class ColdBlooded(Fact):
    """ColdBlooded: True or False"""
    pass


class Parts(Fact):
    """parts: wings or gill or None"""
    pass


class AnimalIdentifier(KnowledgeEngine):
    """
    given some declarations to classify an animal into the following categories:
        protozoa: a group of single-celled microscopic animals
        invertebrate: an animal lacking a backbone.
        fish: a limbless cold-blooded vertebrate animal with gills and fins and living wholly in water.
        bird: a warm-blooded egg-laying vertebrate distinguished by the possession of feathers, wings,
            and a beak and typically by being able to fly.
        mammal: a warm-blooded vertebrate animal, distinguished by the possession of hair or fur,
            the secretion of milk by females for the nourishment of the young
        unknown:
            the rest under this expert system

    Presumption:
        the animal kingdom

    Rules for the Expert System:
        single-celled
            --> True
                -->  protozoa
            --> False
                vertebrate
                    --> False
                        --> invertebrate
                    -->True
                        cold-blooded
                            --> True
                                 gill
                                    --> True
                                        --> fish
                                    --> False
                                        --> unknown
                            --> False
                                --> feather and wings --> bird
                                --> fur and produce milk --> mammal
                                --> else unknown

    Note:
        the rules defined above may not be 100% scientific valid, but the expert system will be implemented
        above rules for learning and demonstration purpose.

    """
    @Rule(SingleCelled(True))
    def is_protozoa(self):
        print('protozoa')

    @Rule(SingleCelled(False), Vertebrate(False))
    def is_invertebrate(self):
        print('invertebrate')

    @Rule(SingleCelled(False), Vertebrate(True), ColdBlooded(True), Parts('gill'))
    def is_fish(self):
        print('fish')

    @Rule(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Parts('wings'), Cover('feather'))
    def is_bird(self):
        print('bird')

    @Rule(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Cover('fur'), ProduceMilk(True))
    def is_mammal(self):
        print('mammal')

    @Rule(
        OR(
            AND(SingleCelled(False), Vertebrate(True), ColdBlooded(True), NOT(Parts('gill'))),
            AND(SingleCelled(False), Vertebrate(True), ColdBlooded(False), NOT(Parts('wings')), Cover('feather')),
            AND(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Parts('wings'), NOT(Cover('feather'))),
            AND(SingleCelled(False), Vertebrate(True), ColdBlooded(False),
                NOT(Parts('wings')), NOT(Cover('feather')), NOT(Cover('fur'))),
            AND(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Cover('fur'), NOT(ProduceMilk(True)))
        )
    )
    def is_unknown(self):
        print('unknown')


"""
expert system 3
"""


class Action(Fact):
    """Action of the Bot"""
    pass


class IndustryChoice(Fact):
    """
    types of Industry:
        A/a: Traditional,
        B/b: Commercial,
        unknown
    """
    pass


class CountryChoice(Fact):
    """"""
    pass


class CompanyChoice(Fact):
    """"""
    pass


def choose_industry(eng):
    choice = input('Which industry you want to talk about, A: Traditional or B: Commercial?')
    if choice == 'A' or choice == 'a':
        eng.declare(IndustryChoice('A'))
    elif choice == 'B' or choice == 'b':
        eng.declare(IndustryChoice('B'))
    else:
        eng.declare(IndustryChoice('unknown'))


def choose_country(eng):
    country = input('Which country you want to know more? Russia, US, China, Or India?')
    if country in ['Russia', 'russia', 'R']:
        eng.declare(CountryChoice('Russia'))
    elif country in ['US', 'us', 'usa', 'USA', 'America', 'america', 'U']:
        eng.declare(CountryChoice('US'))
    elif country in ['China', 'china', 'CN', 'C']:
        eng.declare(CountryChoice('China'))
    elif country in ['India', 'india', 'I']:
        eng.declare(CountryChoice('India'))
    else:
        eng.declare(CountryChoice('unknown'))


def choose_company(eng):
    choice = input("which company you wan to explore more, S: SpaceX, B: Blue Origin, and V: Virgin Galactic?")
    if choice in ['S', 's', 'SpaceX', 'spacex']:
        eng.declare(CompanyChoice('SpaceX'))
    elif choice in ['B', 'b', 'Blue Origin', 'blue origin']:
        eng.declare(CompanyChoice('Blue Origin'))
    elif choice in ['V', 'v', 'Virgin Galactic', 'virgin galactic']:
        eng.declare(CompanyChoice('Virgin Galactic'))
    else:
        eng.declare(CompanyChoice('unknown'))


class SpaceBot(KnowledgeEngine):
    """
    An interactive exert system to talk about the Space Industry.

    Note that this system is built for learning purpose.

    The logic for the simple bot is:

    system_start
            --> get_name
                    --> choose_industry
                                    --> A: Traditional Space Industry
                                                    --> choose_country
                                                            --> Russia
                                                                    --> last_question
                                                            --> US
                                                                    --> last_question
                                                            --> India
                                                                    --> last_question
                                                            --> China
                                                                    --> last_question
                                    --> B: Commercial Space Industry
                                                    --> choose_company
                                                            -->SpaceX
                                                                    --> last_question
                                                            -->Blue Origin
                                                                    --> last_question
                                                            -->Virgin Galactic
                                                                    --> last_question


    last_question
        --> system_start
        --> system_exit

    As shown above, all branches lead to last_question where user can choose to start a new conversation
    or exit the system.

    """
    @DefFacts()
    def _initial_action(self):
        """
        define initial facts to trigger initial actions from the bot
        :return:
        """
        yield Action('greet')

    @Rule(AS.greet << Action('greet'))
    def greet(self, greet):
        """
        first action of the bot:
        1. greeting message
        :return:
        """
        print()
        self.retract(greet)
        print("===========================================================")
        print("=============SpaceBot, V1, By Justin Hu====================")
        print("===========================================================")
        print()
        print("Hello hello, I am your lovely SpaceBot S~Bot.")
        self.declare(Action('ask_name'))

    @Rule(AS.ask_name << Action('ask_name'))
    def ask_name(self, ask_name):
        self.retract(ask_name)
        print()
        guest_name = input("What's your name?")
        print(f'Hi {guest_name}, nice to meet you!')
        self.declare(Fact(name=guest_name))
        self.declare(Action('ask_industry'))

    @Rule(AS.ask_industry << Action('ask_industry'), Fact(name=MATCH.name))
    def ask_industry(self, name, ask_industry):
        self.retract(ask_industry)
        print()
        print(f"Alright, {name}. Let's talk about Space Industry since I am a space bot. Lol.")
        print('I have the knowledge of:')
        print('A. Traditional Space Industry')
        print('B. Commercial Space Industry')
        choose_industry(self)

    @Rule(AS.choice << IndustryChoice('unknown'), Fact(name=MATCH.name))
    def unknown_industry_choice(self, name, choice):
        print()
        self.retract(choice)
        print(f"Sorry {name}, I can't understand your choice.")
        choose_industry(self)

    @Rule(AS.choice << IndustryChoice('A'), Fact(name=MATCH.name))
    def traditional_space_industry(self, name, choice):
        print()
        self.retract(choice)
        print(f'Humans have always been fascinated by space since prehistory, {name}.')
        print('But the modern space industry began only after World War II.')
        print('In October 4th, 1957, the first artificial satellite was launched to the space by USSR.')
        print('In April 12th, 1961, Yuri Gagarin became the first human to journey into space '
              'in the Vostok 1 mission by USSR.')
        print('In December 21th, 1968, the US completed the fist piloted orbital mission of moon: Apollo 8')
        print('In July 20th, 1969, Neil Armstrong(US) became the first human on moon '
              'and took the first sample from moon in the Apollo 11 mission.')
        print('As you can see, Space industry traditionally is dominated by governments, mainly USSR/Russia and US.')
        print('In recent decades, emerging countries including China and India are catching up quickly.\n')

        choose_country(self)

    @Rule(AS.choice << CountryChoice('unknown'), Fact(name=MATCH.name))
    def unknown_country_choice(self, name, choice):
        self.retract(choice)
        print()
        print(f"Sorry {name}, I can't understand your choice.")
        choose_country(self)

    @Rule(AS.choice << CountryChoice('Russia'), Fact(name=MATCH.name))
    def detail_russia(self, name, choice):
        print()
        self.retract(choice)
        print(f'Happy to talk about Russian space industry and development with you, {name}')
        print("Russia's space industry comprises more than 100 companies and employs 250,000 people. \n"
              "Most of the companies are descendants of Soviet design bureaux and state production companies. \n"
              "The industry entered a deep crisis following the dissolution of the Soviet Union, with its fullest \n"
              "effect occurring in the last years of the 1990s. Funding of the space program declined by 80% \n"
              "and the industry lost a large part of its work force before recovery began in the early 2000s. \n"
              "Many companies survived by creating joint-ventures with foreign firms and marketing \n"
              "their products abroad.")

        print("The largest company of Russia's space industry is RKK Energiya. It is the country's main human \n"
              "spaceflight contractor, the lead developer of the Soyuz-TMA and Progress spacecraft and the \n"
              "Russian end of the International Space Station. It employs around 22,000-30,000 people. \n"
              "Progress State Research and Production Rocket Space Center (TsSKB Progress) is the developer \n"
              "and producer of the famous Soyuz launch vehicle. The Soyuz-FG version is used to launch manned \n"
              "spacecraft, while the international joint-venture Starsem markets commercial satellite launches \n"
              "on the other versions. TsSKB Progress is currently leading the development of a new launcher \n"
              "called Rus-M, which is to replace the Soyuz. Moscow-based Khrunichev State Research and Production \n"
              "Space Center is one of the commercially most successful companies of the space industry. It is the \n"
              " developer of the Proton-M rocket and the Fregat upper stage. The company's new Angara rocket family \n"
              " is expected to be put into service 2013. The largest satellite manufacturer in Russia is ISS \n"
              "Reshetnev (formerly called NPO PM). It is main contractor for the GLONASS satellite navigation \n"
              "program and produces the Ekspress series of communications satellites. The company is located \n"
              " in Zheleznogorsk, Krasnoyarsk Krai, and employs around 6,500 people. The leading rocket engine \n"
              "company is NPO Energomash, designer and producer of the famous RD-180 engine. In electric spacecraft \n"
              "propulsion, OKB Fakel, located in Kaliningrad Oblast, is one of the top companies. NPO Lavochkin \n"
              "is Russia's main planetary probe designer. It is responsible for the high-profile Fobos-Grunt \n"
              "mission, Russia's first attempt at an interplanetary probe since Mars 96.")

        self.declare(Action('last_question'))

    @Rule(AS.choice << CountryChoice('US'),  Fact(name=MATCH.name))
    def detail_usa(self, name, choice):
        print()
        self.retract(choice)
        print(f"Happy to talk about US's space industry and development with you, {name}")

        print('The 1957 launch of Sputnik and subsequent Russian firsts in space convinced many U.S. policymakers \n'
              'that the country had fallen dangerously behind its Cold War rival. Consecutive U.S. administrations \n'
              'invested in education and scientific research to meet the Soviet challenge. These investments \n'
              ' propelled the United States to victory in the so-called space race and planted the seeds for \n'
              ' future innovation and economic competitiveness, experts say. Yet, since the 1990s, NASA’s share \n'
              ' of federal spending has waned. The U.S. private sector has ramped up investment in space, and \n'
              ' in May 2020, astronauts launched from U.S. soil for the first time in nearly a decade on a rocket \n'
              ' built by the company SpaceX.')

        print("Due to the Space Shuttle’s retirement in 2011, NASA did not have the means to send astronauts into \n"
              "space by itself for nearly a decade. U.S. astronauts have had to ride Russia’s Soyuz capsule to \n"
              "the ISS—at a cost of up to $82 million per seat. In 2010, former Apollo astronauts Neil Armstrong \n"
              " and Eugene Cernan warned that U.S. leadership in space exploration could suffer. Such criticisms \n"
              ", as well as Trump’s stated desire to land astronauts on the moon during his tenure, spurred \n"
              "the president to boost his budget requests for the agency. ")

        self.declare(Action('last_question'))

    @Rule(AS.choice << CountryChoice('China'), Fact(name=MATCH.name))
    def detail_china(self, name, choice):
        print()
        self.retract(choice)
        print(f"Happy to talk about China's space industry and development with you, {name}")

        print("The space program of the People's Republic of China is directed by the \n"
              "China National Space Administration (CNSA). Its technological roots can be traced back to the late \n"
              "1950s, when China began a ballistic missile program in response to perceived American \n"
              "(and, later, Soviet) threats. However, the first Chinese crewed space program only began \n"
              "several decades later, when an accelerated program of technological development culminated \n"
              "in Yang Liwei's successful 2003 flight aboard Shenzhou 5. This achievement made China the \n"
              "third country to independently send humans into space. Plans currently include a permanent \n"
              "Chinese space station by the end of 2022, crewed expeditions to the Moon and interplanetary \n"
              "missions to explore the Solar System and beyond. Chinese officials have articulated long term \n"
              "ambitions to exploit Earth-Moon space for industrial development and announced China's first \n"
              "landing of a reusable space vehicle at Lop Nur on September 6, 2020")

        print("Initially, the space program of the PRC was organized under the People's Liberation Army, particularly\n"
              "the Second Artillery Corps. In the 1990s, the PRC reorganized the space program as part of a general \n"
              " reorganization of the defense industry to make it resemble Western defense procurement. The China \n"
              "National Space Administration, an agency within the Commission of Science, Technology and Industry \n"
              " for National Defense currently headed by Zhang Kejian, is now responsible for launches. The Long \n"
              "March rocket is produced by the China Academy of Launch Vehicle Technology, and satellites are \n"
              "produced by the China Aerospace Science and Technology Corporation. The latter organizations are \n"
              "state-owned enterprises; however, it is the intent of the PRC government that they should not \n"
              "be actively state-managed and that they should behave as independent design bureaus.")

        self.declare(Action('last_question'))

    @Rule(AS.choice << CountryChoice('India'), Fact(name=MATCH.name))
    def detail_india(self, name, choice):
        print()
        self.retract(choice)
        print(f"Happy to talk about India's space industry and development with you, {name}")

        print("India's space industry is predominantly driven by the national Indian Space Research Organisation \n"
              "(ISRO).The industry includes over 500 private suppliers and other various bodies of the \n"
              "Department of Space in all commercial, research and arbitrary regards.There are relatively \n"
              "few independent private agencies, though they have been gaining an increased role since the \n"
              " start of the 21st century. "
              "In 2019, the space industry of India accounted for $7 billion or 2% of the global space industry \n"
              "and employed more than 45,000 people. Antrix Corporation expects the industry to grow up to $50 \n"
              "billion by 2024 if provided with appropriate policy support.In 2021, the Government of India launched \n"
              "the Indian Space Association to open the Indian space industry to private sectors and start-ups.\n"
              "Several private companies like Larsen & Toubro, Nelco (Tata Group), OneWeb, MapmyIndia, \n"
              "Walchandnagar Industries are founding members of this organisation.Lieutenant General Anil Kumar \n "
              "Bhatt was appointed as the Director General of ISpA")

        print("The Government of India forayed into space exploration when scientists started to launch sounding \n"
              "rockets from Thumba Equatorial Rocket Launching Station (TERLS), Kerala. The establishment of \n"
              "the space agency lead to the development of small launch vehicles SLV-3 and ASLV, \n"
              "followed by larger PSLV and GSLV rockets in the 90s, which allowed India to shift larger \n"
              "payloads and undertake commercial launches for the international market. Private firms started \n"
              " to emerge later as subcontractors for various rocket and satellite components. Reforms liberalising\n"
              " the space sector and nondisclosure agreements came in the late 2010s, leading to the emergence\n"
              " of various private spaceflight companies. By 2019, India had launched more than 300 satellites for \n"
              "various foreign states.[10] There were more than 40 startups in India in early 2021 in various \n"
              "stages of developing their own launch vehicles, designing satellites and other allied activities.\n")

        self.declare(Action('last_question'))

    @Rule(AS.choice << IndustryChoice('B'), Fact(name=MATCH.name))
    def commercial_space_industry(self, name, choice):
        print()
        self.retract(choice)
        print(f"You know what {name}, Before 2012, only vehicles operated by governments had ever visited the ISS. \n "
              "The Dragon by SpaceX was the first commercial vehicle to dock with the station. The milestone was a \n"
              "crowning achievement for the commercial industry, which has permanently altered the spaceflight\n"
              " sector over the last 10 years.")
        print("This decade, the space industry has seen a shift in the way it does business, with newer \n"
              "players looking to capitalize on different markets and more ambitious projects. The result\n"
              " has been an explosion of growth within the commercial sector.")
        print("Most known companies in this domain are S: SpaceX, B: Blue Origin, and V: Virgin Galactic.\n")
        choose_company(self)

    @Rule(AS.choice << CompanyChoice('unknown'), Fact(name=MATCH.name))
    def unknown_company_choice(self, name, choice):
        print()
        self.retract(choice)
        print(f"Sorry {name}, I can't understand your choice.")
        choose_company(self)

    @Rule(AS.choice << CompanyChoice('SpaceX'), Fact(name=MATCH.name))
    def detail_spacex(self, name, choice):
        print()
        self.retract(choice)
        print(f"Happy to talk more about SpaceX with you { name }")

        print("Space Exploration Technologies Corp. (doing business as SpaceX) is an American aerospace \n"
              "manufacturer space transportation services and communications corporation headquartered \n"
              "in Hawthorne, California. SpaceX was founded in 2002 by Elon Musk with the goal of reducing\n"
              " space transportation costs to enable the colonization of Mars. SpaceX manufactures the \n"
              "Falcon 9 and Falcon Heavy launch vehicles, several rocket engines, Cargo Dragon, crew \n"
              "spacecraft and Starlink communications satellites.")

        print("SpaceX's achievements include the first privately funded liquid-propellant rocket to reach \n"
              "orbit around Earth, the first private company to successfully launch, orbit, and recover a\n"
              " spacecraft, the first private company to send a spacecraft to the International Space Station,\n"
              " the first vertical take-off and vertical propulsive landing for an orbital rocket, the first\n"
              " reuse of an orbital rocket, and the first private company to send astronauts to orbit and to\n"
              " the International Space Station. SpaceX has flown the Falcon 9 series of rockets over one \n"
              "hundred times.")

        print("SpaceX is developing a satellite internet constellation named Starlink to provide commercial \n"
              "internet service. In January 2020, the Starlink constellation became the largest satellite \n"
              "constellation ever launched. The company is also developing Starship, a privately funded, \n"
              "fully reusable, super heavy-lift launch system for interplanetary spaceflight. Starship is \n"
              "intended to become SpaceX's primary orbital vehicle once operational, supplanting the \n"
              "existing Falcon 9, Falcon Heavy, and Dragon fleet. Starship will have the highest payload\n"
              " capacity of any orbital rocket ever on its debut, scheduled for the early 2020s.")

        self.declare(Action('last_question'))

    @Rule(AS.choice << CompanyChoice('Blue Origin'), Fact(name=MATCH.name))
    def detail_blue_origin(self, name, choice):
        print()
        self.retract(choice)
        print(f"Happy to talk more about Blue Origin with you { name }")

        print("Blue Origin, LLC is an American privately funded aerospace manufacturer and sub-orbital \n"
              "spaceflight services company headquartered in Kent, Washington.Founded in 2000 by Jeff Bezos, \n"
              "the founder and executive chairman of Amazon, the company is led by CEO Bob Smith and \n"
              "aims to make access to space cheaper and more reliable through reusable launch vehicles.\n"
              "Rob Meyerson led Blue Origin from 2003 to 2017 and served as its first president.\n "
              "Blue Origin is employing an incremental approach from suborbital to orbital flight,\n"
              "with each developmental step building on its prior work. The company's name refers to \n"
              "the blue planet, Earth, as the point of origin.")

        print("Blue Origin moved into the orbital spaceflight technology development business in 2014, \n"
              "initially as a rocket engine supplier for others via a contractual agreement to build a new \n"
              "large rocket engine, the BE-4, for major US launch system operator United Launch Alliance (ULA).\n"
              " Blue said the BE-4 would be 'ready for flight' by 2017. By 2015, Blue Origin had announced \n"
              "plans to also manufacture and fly its own orbital launch vehicle, known as the New Glenn,\n "
              "from the Florida Space Coast. BE-4 had been expected to complete engine qualification\n "
              "testing by late 2018.[18] However, by August 2021, the flight engines for ULA have\n "
              "still not been qualified, and Ars Technica revealed in an in-depth article serious \n"
              "technical and managerial problems in the BE-4 program.")

        print("In May 2019, Jeff Bezos unveiled Blue Origin's vision for space and also plans for a moon lander \n"
              "known as 'Blue Moon', set to be ready by 2024.[20] On July 20, 2021, Blue Origin sent its first \n"
              "crewed mission into space via its New Shepard rocket and spaceflight system. The flight was\n "
              "approximately 10 minutes, and crossed the Kármán Line. Blue Origin founder Jeff Bezos was\n "
              "part of the four member crew along with his brother Mark Bezos, Wally Funk, and Oliver Daemen\n")

        self.declare(Action('last_question'))

    @Rule(AS.choice << CompanyChoice('Virgin Galactic'), Fact(name=MATCH.name))
    def detail_virgin_galactic(self, name, choice):
        print()
        self.retract(choice)
        print(f"Happy to talk more about Virgin Galactic with you { name }")

        print("Virgin Galactic is an American spaceflight company founded by Richard Branson and his \n"
              "British Virgin Group retains an 11.9% stake through Virgin Investments Limited. It is headquartered\n"
              " in California, USA, and operates from New Mexico. The company is developing commercial \n"
              "spacecraft and aims to provide suborbital spaceflights to space tourists. Virgin Galactic's \n"
              "suborbital spacecraft are air launched from beneath a carrier airplane known as White Knight Two. \n"
              "Virgin Galactic‘s maiden spaceflight occurred in 2018 with its VSS Unity spaceship.\n"
              "Branson had originally hoped to see a maiden spaceflight by 2010, but the date was delayed for \n"
              "several years, primarily due to the October 2014 crash of VSS Enterprise")

        print("On 13 December 2018, VSS Unity achieved the project's first suborbital space flight, VSS \n"
              "Unity VP-03, with two pilots, reaching an altitude of 82.7 kilometres (51.4 mi), and officially \n"
              "entering outer space by U.S. standards. In February 2019, the project carried three people,\n "
              "including a passenger, on VSS Unity VF-01, with a member of the team floating within the \n"
              "cabin during a spaceflight that reached 89.9 kilometres (55.9 mi).")

        print("On 11 July 2021, the company founder Richard Branson and three other employees rode on a flight\n"
              " as passengers, marking the first time a spaceflight company founder has travelled on his own\n"
              " ship into outer space (according to the NASA definition of outer space beginning at 50 miles\n"
              " above the Earth)")

        self.declare(Action('last_question'))

    @Rule(AS.last_question << Action('last_question'), Fact(name=MATCH.name))
    def last_question(self, name, last_question):
        print()
        self.retract(last_question)
        print(f"Alright, we've talked so much. Very excited to talk to you {name}.")
        print('Do you want to A. end the conservation and say goodbye or B. start a new talk?')
        choice = input('Please choose A or B?')
        if choice in ['A', 'a']:
            self.declare(Action('bye'))
        elif choice in ['B', 'b']:
            self.declare(Action('greet'))
        else:
            self.declare(Action('bye'))

    @Rule(AS.bye << Action('bye'), Fact(name=MATCH.name))
    def good_bye(self, name, bye):
        print()
        self.retract(bye)
        print(f'Very nice to chat with you, {name}.')
        print('Hope you enjoy the conversation as I do.')
        print('Bye Bye!')


def test_es1():
    # Test ES1: AnimalChecker
    print('Testing ES1:AnimalChecker')
    es1 = AnimalChecker()
    es1.reset()
    es1.declare(Cover('feathers'), Wings(True))
    print('an animal with feathers and wings is')
    es1.run()

    es1.reset()
    es1.declare(Cover('fur'), Wings(True))
    print('an animal with fur and wings is')
    es1.run()

    es1.reset()
    es1.declare(Cover('feathers'), Wings(False))
    print('an animal with feathers and no wings is')
    es1.run()

    es1.reset()
    es1.declare(Cover('fur'), Wings(False))
    print('an animal with fur and no wings is')
    es1.run()


def test_es2():
    # Test ES2: AnimalIdentifier
    print('Testing ES2:AnimalIdentifier')
    es2 = AnimalIdentifier()
    es2.reset()
    es2.declare(SingleCelled(True))
    print('an animal with single cell is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(False))
    print('an animal that is invertebrate is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(True), Parts('gill'))
    print('an animal that is multi-celled, vertebrate and coldblooded with gills is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Parts('wings'), Cover('feather'))
    print('an animal that is multi-celled, vertebrate and warmblooded with feather and wings is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Cover('fur'), ProduceMilk(True))
    print('an animal that is multi-celled, vertebrate and warmblooded with fur and can produce milk is')
    es2.run()

    # unknown cases
    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(True), Parts('None'))
    print('an animal that is multi-celled, vertebrate and coldblooded without gills is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(True), Parts('wings'))
    print('an animal that is multi-celled, vertebrate and coldblooded with wings is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Parts('None'), Cover('feather'))
    print('an animal that is multi-celled, vertebrate and warmblooded with feather and no wings is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Parts('wings'), Cover('None'))
    print('an animal that is multi-celled, vertebrate and warmblooded with wings and no feather is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Parts('None'), Cover('None'))
    print('an animal that is multi-celled, vertebrate and warmblooded with no wings and no feather is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Cover('None'), ProduceMilk(True))
    print('an animal that is multi-celled, vertebrate and warmblooded with no cover and can produce milk is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Cover('fur'), ProduceMilk(False))
    print('an animal that is multi-celled, vertebrate and warmblooded with fur and can not produce milk is')
    es2.run()

    es2.reset()
    es2.declare(SingleCelled(False), Vertebrate(True), ColdBlooded(False), Cover('None'), ProduceMilk(False))
    print('an animal that is multi-celled, vertebrate and warmblooded with no fur and can not produce milk is')
    es2.run()


def test_es3():
    """
    test expert system 3: SpaceBot
    :return:
    """
    es3 = SpaceBot()
    es3.reset()
    es3.run()


if __name__ == '__main__':
    """
    Test 3 ES systems:  AnimalChecker, AnimalIdentifier, SpaceBot
    Choose which test to run by commenting in/out different testing functions
    """
    # test_es1()
    # test_es2()
    test_es3()
