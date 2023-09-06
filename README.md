# Date-Finder

This code is based on an assignment from the University of Michigan's Coursera course titled Appled Text Mining in Python.

This code is intended to demonstrate a nontrivial use of RegEx for text data extraction.

The code looks through a dataset of 500 typed medical notes and extracts dates from each.

The formatting of the dates in the notes varies quite a bit. For example, dates could be written 08/01/2023; 8/1/23; August 1, 2013; Aug 1, 2013; Aug. 1, 2013; 1 Aug 2013; Aug 2013 (no day); 2023 (no day or month), etc.

Assumptions: dates missing a day will be assigned the first day of the month, 01. Dates missing a day and month will be assigned January first of the given year.
