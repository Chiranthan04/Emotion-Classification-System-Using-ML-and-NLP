import sqlite3
from datetime import datetime

# Database connection
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

# Page Visit Tracking Functions
def create_page_visited_table():
    c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT, timeOfvisit TIMESTAMP)')
    conn.commit()

def add_page_visited_details(pagename, timeOfvisit):
    c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', (pagename, timeOfvisit))
    conn.commit()

def view_all_page_visited_details():
    c.execute('SELECT * FROM pageTrackTable')
    data = c.fetchall()
    return data

# Prediction Tracking Functions
def create_prediction_table():
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, model_used TEXT,probability NUMBER, timeOfvisit TIMESTAMP)')
   
    conn.commit()

def add_prediction_details(rawtext, prediction, model_used, probability, timeOfvisit):
  
    c.execute('INSERT INTO emotionclfTable(rawtext, prediction, model_used, probability, timeOfvisit) VALUES (?, ?, ?, ?, ?)',
              (rawtext, prediction, model_used, probability, timeOfvisit))
    conn.commit()

def view_all_prediction_details():
    c.execute('SELECT * FROM emotionclfTable')
    data = c.fetchall()
    return data

# Initialize Tables
def setup_tables():
    create_page_visited_table()
    create_prediction_table()

setup_tables()
