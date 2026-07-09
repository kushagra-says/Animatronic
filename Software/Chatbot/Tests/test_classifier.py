"""
Tests the Classifier module.
"""

from classifier import Classifier


def main():

    classifier = Classifier()

    print("=" * 60)
    print("CLASSIFIER TEST")
    print("=" * 60)

    examples = [

        "Hello!",

        "My favorite band is Tame Impala.",

        "I study AI & ML.",

        "What is recursion?",

        "My name is Kushagra.",

        "I am building an offline AI."
    ]

    for sentence in examples:

        print("\n" + "-" * 60)
        print(f"Input : {sentence}")

        result = classifier.classify(sentence)

        print(f"Remember : {result['remember']}")
        print(f"Type      : {result['type']}")
        print(f"Reason    : {result['reason']}")


if __name__ == "__main__":
    main()