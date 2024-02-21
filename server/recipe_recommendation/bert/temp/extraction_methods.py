# Import the spacy module and load the English model
import spacy
nlp = spacy.load("en_core_web_sm")

# Define a list of ingredients
ingredients = ['Potato starch (for dusting cake pan)', '2 teaspoons vegetable oil', '1/2 cup matzoh cake meal', '3/4 cup potato starch', '8 extra-large eggs, separated, at room temperature', '1 cup sugar', '1/4 cup orange juice', 'Juice of 1 large lemon', '1 teaspoon freshly grated orange zest', '1 teaspoon freshly grated lemon zest', '1 1/2 teaspoons pure vanilla extract', '1/2 teaspoon almond extract', '1/4 teaspoon salt', '3 pints strawberries, stemmed, washed, and thinly sliced', '1/2 cup orange juice', '1 tablespoon sugar']

# Loop through the ingredients and apply the NLP model
for ingredient in ingredients:
  # Parse the ingredient with spacy
  doc = nlp(ingredient)
  # Loop through the noun phrases in the doc
  for np in doc.noun_chunks:
    # Print the noun phrase
    print(np)


print("===============================================")
# Import the re module
import re
# Define a regular expression pattern to match the amounts and measurements
pattern = r"^\d+\/?\d*\s*[a-zA-Z]*\s*"

# Loop through the ingredients and apply the pattern
for ingredient in ingredients:
  print('ingredient:', ingredient)
  # Replace the matched text with nothing
  ingredient = re.sub(pattern, "", ingredient)
  print('ingredient2:', ingredient)
  # Print the result
  print(ingredient)