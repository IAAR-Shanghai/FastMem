import json
import pandas as pd
import re

def data_address(name, path=None, mode="test"):
    clean_data = []
    if name == "cnndm":
        # Download: https://github.com/artmatsak/cnn-dailymail/tree/b6d20708a1180f58dd96b5ab923ed099ced6b2ab
        with open("./cnn-dailymail/cnn_dm/{}.source".format(mode), "r") as r1:
            with open("./cnn-dailymail/cnn_dm/{}.target".format(mode), "r") as r2:
                for nline, (line1, line2) in enumerate(zip(r1, r2)):
                    clean_data.append({
                        "article": line1.strip(),
                        "summary": line2.strip()
                    })

    elif name == "wikihow":
        # Download: github https://github.com/mahnazkoupaee/WikiHow-Dataset?tab=readme-ov-file
        # Download the first dataset wiki-ALL and concatenate each paragraph as a sample to generate a summary into a list
        Data = pd.read_csv(r'{}'.format(path))
        Data = Data.astype(str)
        rows, columns = Data.shape
        with open("./WikiHow-Dataset/all_test.txt", "r") as r2:
            test_data = r2.readlines()

        for row in range(rows):
            abstract = Data.loc[row,'headline']      # headline is the column representing the summary sentences
            article = Data.loc[row,'text']           # text is the column representing the article

            #  a threshold is used to remove short articles with long summaries as well as articles with no summary
            if len(abstract) < (0.75*len(article)):
                # remove extra commas in abstracts
                abstract = abstract.replace(".,",".")
                abstract = abstract
                # remove extra commas in articles
                article = re.sub(r'[.]+[\n]+[,]',".\n", article)
                article = article


                # a temporary file is created to initially write the summary, it is later used to separate the sentences of the summary
                with open('./WikiHow-Dataset/temporaryFile.txt', 'w') as t:
                    t.write(abstract)

                # file names are created using the alphanumeric charachters from the article titles.
                # they are stored in a separate text file.
                filename = Data.loc[row,'title']

                summary_seq = ""
                with open('./WikiHow-Dataset/temporaryFile.txt', 'r') as t:
                    for line in t:
                        line=line.lower()
                        if line != "\n" and line != "\t" and line != " ":
                            summary_seq += line
                title = "".join(filename.split(' ')) + '\n'
                if title in test_data:
                    clean_data.append({
                        "title": filename,
                        "summary": summary_seq,
                        "article": article
                    })

    elif name == "xsum":
        # Download: https://huggingface.co/datasets/EdinburghNLP/xsum
        """
        {
            "article": ,
            "abstract": ,
            "candidates": ,
            "article_untok": ,
            "abstract_untok": ,
            "candidates_untok": ,
        }
        """
        from datasets import load_dataset
        # 加载测试集
        test_dataset = load_dataset('xsum', split='test', trust_remote_code=True)
        for line in test_dataset:
            clean_data.append({
                "id": line['id'],
                "text": line['document'],
                "summary": line['summary'],
            })
    print(name, len(clean_data))
    with open("{}_addressed_{}.jsonl".format(name, mode), "w") as w:
        for item in clean_data:
            w.write(json.dumps(item, ensure_ascii=False))
            w.write("\n")

if __name__ == '__main__':
    # wikihow
    name = "wikihow"
    path = "./WikiHow-Dataset/wikihowAll.csv"
    data_address(name, path, mode="train")
    # xsum
    name = "xsum"
    path = ""
    data_address(name, path, mode="train")
    # cnndm
    name = "cnndm"
    path = ""
    data_address(name, path, mode="train")

