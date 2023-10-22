This code is the L2 Syntactic Complexity Analyzer described in:

Lu, Xiaofei (2010). Automatic analysis of syntactic complexity in second language writing. International Journal of Corpus Linguistics, 15(4):474-496.

Version 3.3.3, released June 30, 2016

Copyright (C) 2016 Xiaofei Lu

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

To download the latest version of this software, follow the appropriate link at
	http://www.personal.psu.edu/xxl13/download.html

==============================================================================
ABOUT

L2 Syntactic Complexity Analyzer is designed to automate syntactic complexity analysis of written English language samples produced by advanced learners of English using fourteen different measures proposed in the second language development literature. The analyzer takes a written English language sample in plain text format as input and generates 14 indices of syntactic complexity of the sample. 

The analyzer is implemented in python and runs on UNIX-like (LINUX, MAC OS, or UNIX) systems with Java 1.5 and python 2.5 or higher installed. The analyzer takes as input a plain text file, counts the frequency of the following 9 structures in the text: words (W), sentences (S), verb phrases (VP), clauses (C), T-units (T), dependent clauses (DC), complex T-units (CT), coordinate phrases (CP), and complex nominals (CN), and computes the following 14 syntactic complexity indices of the text: mean length of sentence (MLS), mean length of T-unit (MLT), mean length of clause (MLC), clauses per sentence (C/S), verb phrases per T-unit (VP/T),, clauses per T-unit (C/T), dependent clauses per clause (DC/C), dependent clauses per T-unit (DC/T), T-units per sentence (T/S), complex T-unit ratio (CT/T), coordinate phrases per T-unit (CP/T), coordinate phrases per clause (CP/C), complex nominals per T-unit (CN/T), and complex nominals per clause (CP/C).

The analyzer calls the Stanford praser (Klein & Manning, 2003) to parse the input file and Tregex (Levy and Andrew, 2006) to query the parse trees. Both the Stanford parser and Tregex are bundled in this download and installation along with the appropriate licenses. 

CONTENTS

[1] Running the single file analyzer
[2] Input format
[3] Output format
[4] Running the multiple file analyzer
[5] A list of the files included in this package

==============================================================================
[1] Running the single file analyzer

To run the single file analyzer, type the following at the command line:

python analyzeText.py <input_file> <output_file>

Note that the python script should be called from within this directory. To make sure everything runs correctly, run the following and compare your output with the sample1_output file in the samples/ subdirectory. 

python analyzeText.py samples/sample1.txt samples/sample1_testing
==============================================================================
[2] Input format

The input file should be a clean English text in plain text format (with a .txt suffix in the filename). Sample files can be found in the "samples" sub-directory.

==============================================================================
[3] Output format

A name of the output file must be provided, but you can name it anything you like. 

The first line in the output file is a comma-delimited list of 24 fields, including Filename, abbreviations of the 9 structures mentioned above, and abbreviations of the 14 syntactic complexity indices mentioned above. 

The second line (and subsequent lines if analyzing multiple files in a directory) is a comma-delimited list of 24 values, including the name of the input file, the frequency counts of the 9 structures, and the values of the 14 syntactic complexity indices. 

This format may be hard to read but allows for easy import to Excel or SPSS. 

==============================================================================
[4] Running the multiple file analyzer

To run the multiple file analyzer, type the following at the command line:

python analyzeFolder.py <path_to_input_file_folder> <output_file>

path_to_input_file_folder is the path to the folder or directory that contains the text files you want to analyze (e.g., /home/inputFiles/). The path should end with a slash, as in the example. Only files that end with the .txt suffix will be analyzed. 

Note that the python script should be called from within this directory. To make sure everything runs correctly, run the following and compare your output with the samples_output file in the samples/ subdirectory. 

python analyzeFolder.py samples/ samples/samples_testing
==============================================================================
[5] A list of the files included in this package

README-L2SCA.txt - this file

analyzeText.py - the single file analyzer

analyzeFolder.py - the multiple file analyzer

samples/ - this directory includes the following sample files:

sample1.txt: an English text in plain text format

sample2.txt: another English text in plain text format

sample1_output: sample output file generated by the single file analyzer

samples_output: sample output file generated by the multiple file analyzer

All files for Tregex 3.3.1

Stanford parser 3.3.1
