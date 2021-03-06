# CYK Parser

This is the work for the TP2 for for the master MVA course Natural Language processing 

## Dependencies

Necessary packages : Python>=3.6, nltk, numpy, pyevalb, pickle.
 
## Usage

There are 2 mode to use the code :

- `--mode dataset -d <path_to_dataset>` to evaluate the parser a dataset in the same format as the sequoia corpus. By default, it is launched on the last 10% of the sequoia corpus.
- `--mode sentence -s "Your sentences to parse"` will run the parser on a unique sentence.  Each token has to be separated by a single whitespace. The input can take several lines with one sentence per line.

There are several other option :
- `-h` : give information on all the different options
- `-w True` : if given, the output of a dataset is written to `<datasetPath>\<dataset_name>_output.txt`. The non parsable sentence from the dataset are written to `non_parsable.txt`.
- `-e <path_to_embedding>`: if given, it will use the given file for Out of Vocabulary embedding. The default embedding is : [polyglot-fr.pkl](https://sites.google.com/site/rmyeid/projects/polyglot)

## Output

If mode dataset : the output is the different metrics to check the accuracy of parsing toward dataset. If `-w` is given, see Usage for the written files.

If mode sentence : the output is the parsed tree of the input sentence. 

## Examples

Parse the sentence "Un ours mange une pomme .":
```bash
./run.sh -s "Un ours mange une pomme ."
```

Get accuracy on the validation dataset :
```bash
./run.sh --mode dataset
```


