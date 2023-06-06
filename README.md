# OMR Sheet Reader and Report Card Generator

This project is designed to read OMR (Optical Mark Recognition) sheets using sacned copy of OMR sheets and generate individual data for each person. It also includes a report card generation feature that takes correct answers and mark weights into account.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

OMR Sheet Reader is a Python-based application that utilizes the OpenCV library to read OMR sheets from photos. It employs image processing techniques to extract answers marked by individuals and generate data corresponding to each person. Additionally, the project includes a report card generation feature that takes the correct answers and mark weights into account, producing individualized report cards for each student.

## Features

- OMR Sheet Reading: The application is capable of processing photos of OMR sheets and extracting the marked answers.
- Data Generation: It generates individual data for each person, including their answered questions and corresponding details.
- Report Card Generation: The project provides functionality to generate report cards based on the correct answers and mark weights.

## Installation

To use this application, you need to have Python and OpenCV installed on your system. Follow the steps below to set up the project:

1. Clone the repository:

```bash
git clone https://github.com/your-username/omr-sheet-reader.git
```

2. Navigate to the project directory:
 
```bash
cd omr-sheet-reader
```

3. Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
python omr_sheet_reader.py
```
# Usage
Once the application is running, follow these steps to utilize its features:

1. Capture or import a photo of the OMR sheet you want to process. 
2. then maka a .xml file with the correct answers. 
3. the you run a bash script that will take each photo and read its data. and generate .csv file with the repsone on the photo. 
4. you can generate the report card of individual peoples uing the googole api

# Contributing 

Contributing to the project are welcome. If you encounter any issues or have ideas for improvements, please open an issue or submit a pull request. 

# Licence
This project is licensed under the MIT License. You are free to modify, distribute, and use the code in any way you see fit. Refer to the LICENSE file for more details.


