#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create a set with curly braces
s1 ={1,6,8,7,4,0,1,9}
print(s1)
print(type(s1))


# In[5]:


lst1 = [1,6,8,7,4,0,1,9]


# In[7]:


s2 = set(lst1)
print(s2)
print(type(s2))


# In[9]:


# Union operation using | operator
s1 = {1,2,3,4}
s2 = {3,4,5,6}


# In[11]:


s1 | s2


# In[13]:


s1.union(s2)


# In[15]:


s1 & s2


# In[17]:


s1.intersection(s2)


# In[19]:


# difference of two sets
s1 = {2,3,5,6,7}
s2 = {5,6,7}


# In[21]:


s1 - s2


# In[23]:


s2 - s1


# In[31]:


# Sysmetric difference
s1={1,2,3,4,5}
s2={4,5,6,7,8}


# In[33]:


s1.symmetric_difference(s2)


# In[35]:


s2.symmetric_difference(s1)


# In[41]:


str1 = "Welcome to aiml class"
print(str1)
str2 = 'We started with python'
print(str2)
str3 = '''This is an awesome class'''
print(str3)


# In[45]:


print(type(str1))
print(type(str2))
print(type(str3))


# In[53]:


str4 = '''He said,It's awesome!"'''
print(str4)


# In[57]:


# slicing in strings
print(str1)
str1[5:10]


# In[61]:


dir(str)


# In[63]:


# Reversing the string
str1[::-1]


# In[65]:


# Use of split()
print(str1)
str1.split()


# In[73]:


# Use of join() method in strings
str4 = "Hi. How are you?"
' '.join(str4)


# In[75]:


reviews = ["The product is awesome", "Great Service"]
joined_string = ' '.join(reviews)
joined_string


# In[77]:


# Use of strip() method
str5 = "   Hello, How are you?   "
str5


# In[79]:


str5.strip()


# In[81]:


str5 = "Hello,How are you?"
str5


# In[83]:


sales_data = {
    "ProductID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "ProductName": ["Laptop", "Mouse", "Keyboard", "Monitor", "Chair", "Desk", "Webcam", "Headphones", "Printer", "Tablet"],
    "Category": ["Electronics", "Accessories", "Accessories", "Electronics", "Furniture", "Furniture", "Electronics", "Accessories", "Electronics", "Electronics"],
    "PriceRange": ["High", "Low", "Low", "Medium", "Medium", "Medium", "Low", "Low", "Medium", "High"],
    "StockAvailable": [15, 100, 75, 20, 10, 8, 50, 60, 25, 12],
}


# In[85]:


for k,v in sales_data.items():
    print(k,set(v), end =',')
    print('/n')


# In[87]:


# Original reviews dictionary
reviews = {
    "Review1": "The product quality is excellent and delivery was prompt. The product functionality is versatile",
    "Review2": "Good service but the packaging could have been better. The customer service has to improve",
    "Review3": "The product works fine, but the customer support is not very helpful. I rate the product as excellent",
}

# Result dictionary to store analysis of reviews
review_analysis = {}

# Process each review
for key, review in reviews.items():
    # Split the review into words
    words = review.lower().replace('.', '').replace(',', '').split()
    # Create a sub-dictionary with word count and unique words
    review_analysis[key] = {
        "WordCount": len(words),
        "UniqueWords": list(set(words))
    }

review_analysis


# In[89]:


d1 = {"Ram": 178, "Hari": 174, "Ramya":126}


# In[91]:


for k in d1.keys():
    print(k)


# In[97]:


for v in d1.values():
    print(v)


# In[99]:


for k,v in d1.items():
    print(k,v)


# In[101]:


d1["Harry"] = 168
d1


# In[ ]:




