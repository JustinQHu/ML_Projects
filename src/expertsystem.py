from experta import *
import schema

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


class InterestBot(KnowledgeEngine):
    """
    An interactive exert system
    """


if __name__ == '__main__':
    """
    Test 3 ES systems:  AnimalChecker, AnimalIdentifier
    """

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
