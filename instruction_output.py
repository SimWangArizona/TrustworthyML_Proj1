from transformers import pipeline

# generator = pipeline('text-generation', model="/workspace/dongweiw/models/OPT/OPT-1.3b")
# output = generator("What are we having for dinner?")
# print(output[0]['generated_text'])
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="opt model to load")

    parser.add_argument("--context", type=str, default="", help="instructions")
    args = parser.parse_args()

    generator = pipeline('text-generation', model=args.model)
    output = generator(args.context)
    print(output[0]['generated_text'])
