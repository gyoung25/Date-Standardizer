'''
This code is based on an assignment from the University of Michigan's Coursera course titled Appled Text Mining in Python.

This code is intended to demonstrate a nontrivial use of RegEx for text data extraction.

The code looks through a dataset of 500 typed medical notes and extracts dates from each.

The formatting of the dates in the notes varies quite a bit. For example, dates could be written 08/01/2023; 8/1/23; August 1, 2013; Aug 1, 2013; Aug. 1, 2013; 1 Aug 2013; Aug 2013 (no day); 2023 (no day or month), etc.

Assumptions: dates missing a day will be assigned the first day of the month, 01. Dates missing a day and month will be assigned January first of the given year.
'''

import pandas as pd
import re

doc = []
with open('date_doc.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)

def date_finder(dates_text):
    """
    Arguments:
        dates_text - a list of texts from which dates will be extracted
    Returns:
        dates - a list of strings of dates extracted from the texts in df
    """
    import re
    order = None
    dates = []
    missed = 0
    found = 0
    for i,line in enumerate(dates_text):
        #catch all dates of the form [M]M-[D]D-YYYY or [M]M-YYYY or [M]M/[D]D/YYYY or [M]M/YYYY
        date_num1 = re.search(r'((0[1-9]|1[012]|[1-9])(\/|-)((0[1-9]|[12][0-9]|3[01]|[1-9])(\/|-))*((19|20)\d{2}))',line)
        #catch all dates of the form [M]M-[D]D-YY or [M]M/[D]D/YY
        date_num2 = re.search(r'((0[1-9]|1[012]|[1-9])(\/|-)((0[1-9]|[12][0-9]|3[01]|[1-9])(\/|-))(\d{2}))',line)
        #catch all dates of the form YYYY
        date_num3 = re.search(r'((19|20)\d{2})', line)
        #catch all dates of the form [D]D Month [YY]YY
        date_str1 = re.search(r'(0[1-9]|[12][0-9]|3[01]|[1-9])\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\.,\w]*\s((19|20)\d{2}|\d{2})',line)
        #catch all dates of the form Month [D]D [YY]YY 
        date_str2 = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\.,\w]*[-\s](0[1-9]|[12][0-9]|3[01]|[1-9])*\w*,*\s*((19|20)\d{2}|\d{2})',line)
        
        # Note: the order of the RegEx searches below matters. For example, if a date is written MM/DD/YYYY, 
        # then both date_num1 and date_num2 would catch it, but date_num2 would not return the full year, only
        # the first two digits of the four-digit year. We must therefore collect all dates of this form using
        # date_num1, then move on the the remaining dates in a numerical format (that is, the ones of the form
        # MM/DD/YY) after date_num1 misses them.
        if date_num1 != None:
            dates.append(date_num1[0])
        elif date_num2 != None:
            if re.search(r'\/\d{2}$',date_num2[0]) != None:
                date_num2 = re.sub(r'\/\d{2}$',"/19"+str(date_num2[0][-2:]),date_num2[0])
                dates.append(date_num2)
            else:
                dates.append(date_num2[0])
        elif date_str1 != None:
            dates.append(date_str1[0])
        elif date_str2 != None:
            dates.append(date_str2[0])
        elif date_num3 != None:
            dates.append(date_num3[0])
        else:
            print(line)
            missed += 1
            print(f'Missed line {i}')
    #manual corrections
    #print(dates[298])
    dates[298] = 'January 1993'
    dates[313] = 'December 1978'
    return dates

def date_sorter(dates):
    """
    Arguments:
        dates - list of strings of dates
    Returns:
        date_series - pandas Series containing a sorted list of dates in standardized format
    """
    # Store the dates in a pandas Series
    dates_series = pd.Series(dates)
    
    # Convert the dates collected into datetime format for consistent formatting and easier analysis
    date_series = pd.to_datetime(dates_series)
    # Sort this list of dates in ascending order
    date_series.sort_values(inplace = True, kind = 'mergesort')
    #print(missed)
    return date_series
'''
dates = date_finder(doc)
date_series = date_sorter(dates)
print(f'The head of the sorted records looks like \n{date_series.head()}')
print(f'and the tail looks like \n{date_series.tail()}')
print(f'The oldest medical record is from {date_series.iloc[0].date()} and the most recent is from {date_series.iloc[-1].date()}.')

with open('dates.txt', 'w') as f:
    for line in dates:
        f.write(f"{line}\n")
'''



