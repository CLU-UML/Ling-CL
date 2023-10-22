This code is the lexical complexity analyzer described in

Lu, Xiaofei (2012). The relationship of lexical richnes to the quality 
of ESL speakers' oral narratives. The Modern Language Journal, 96(2), 190-208. 

Version 1.1 Released on February 12, 2013

Copyright (C) 2013 Xiaofei Lu
 
This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
details.

You should have received a copy of the GNU General Public License along with 
this program; if not, write to the Free Software Foundation, Inc., 59 Temple 
Place, Suite 330, Boston, MA  02111-1307  USA

To download the latest version of this software, follow the appropriate link 
at
        http://www.personal.psu.edu/xxl13/download.html


1. About

This tool computes the lexical complexity of English texts using 25 different 
measures. Information on the measures can be found in Lu (2012). This 
tool uses frequency lists derived from the British National Corpus and the
American National Corpus. 

2. Running the tool

2.1 Input files: All input files must be POS-tagged and lemmatized first and 
must be in the following format (see files in the samples folder for 
examples). The file should contain a minumum of 50 words. 

lemma_pos lemma_pos lemma_pos ...

or 

lemma_pos
lemma_pos
lemma_pos

You can use any POS tagger and lemmatizer, as long as the Penn Treebank POS 
tagset is adopted and the input file is appropriately formatted. In Lu 
(2012), the following POS tagger and lemmaitzer were used:

The Stanford POS tagger: 
http://nlp.stanford.edu/software/tagger.shtml

MORPHA: 
http://www.informatics.susx.ac.uk/research/groups/nlp/carroll/morph.html

2.2 Analyzing a single file: To get the lexical complexity of a single file, 
run the following from this directory. Replace input_file with the actual 
name of your input file and output_file with the desired name of your output 
file.

python lc.py input_file > output_file

e.g.,

python lc.py samples/1.lem > 1.lex

To use the American National Corpus (ANC) wordlist instead of the BNC wordlist
for lexical sophistication analysis, use the lc-anc.py script, e.g.,

python lc-anc.py samples/1.lem > 1-anc.lex

2.3 Analyzing multiple files: To get the lexical complexity of two or more 
files within a single folder, run the following from this directory. Replace 
path_to_folder with the actual path to the folder that contains your files 
and output_file with the desired name of your output file. The folder should 
only contain the files you want to analyze.

python folder-lc.py path_to_folder > output_file

e.g.,

python folder-lc.py samples/ > samples.lex

To use the American National Corpus (ANC) wordlist instead of the BNC wordlist
for lexical sophistication analysis, use the folder-lc-anc.py script, e.g.,

python folder-lc-anc.py samples/ > samples-anc.lex

2.4 Using the output: The output file is comma-delimited and can be loaded to 
excel and spss directly for analysis.
