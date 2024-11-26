from sentence_transformers import SentenceTransformer
import chromadb
import json
import csv
import google.generativeai as genai
import textwrap
import os
import json
import logging
from secretsIn import API_KEY
# from flask import Flask, render_template, request, jsonify
logging.basicConfig(level=logging.DEBUG)


# app = Flask(__name__)


client = chromadb.Client()
# client = chromadb.PersistentClient(path="/path/to/save/to")

collection = client.create_collection('movie_collection')

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('results.json') as file:
    results = json.load(file)
    for result in results:
        result = result['args']
        embedding = model.encode(json.dumps(result))
        collection.add(
            embeddings=embedding,
            metadatas=[{"text": json.dumps(result)}],
            ids=[result['id']]
        )
        # print(f"Added document with id {result['id']}")
        # print(f"Document added to the collection with id {result['id']}")


def generate_prompt(id, movie_name, plot, genre, cast, directors, music, release_year):
    prompt = f"Id: {id}\nMovie Name: {movie_name}\nPlot: {plot}\nGenre: {genre}\nCast: {cast}\nDirectors: {directors}\nMusic: {music}\nRelease Year: {release_year}"
    return prompt


def call_gemini(prompt: str):
    genai.configure(api_key=API_KEY)
    production_obj = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'music':  genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type=genai.protos.Type.STRING)),
            'production_house':  genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type=genai.protos.Type.STRING))
        }
    )

    movie_object = genai.protos.FunctionDeclaration(
        name='get_json_from_data_set',
        description=textwrap.dedent("""\
            Extracts Json data from the Movie Database
            """),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'id': genai.protos.Schema(type=genai.protos.Type.STRING, description="The unique id of the Movie"),
                'name': genai.protos.Schema(type=genai.protos.Type.STRING, description="The name of the Movie from the given source string"),
                'plot': genai.protos.Schema(type=genai.protos.Type.STRING, description="Plot of the Movie from the given source string"),
                'genres': genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING, description="The list of genres of the Movie from the given source string")),
                'cast': genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING, description="The list of cast acted in the movie from the given source string")),
                'directors': genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING, description="The list of directors of the movie from the given source string")),
                'production': genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=production_obj),
                'release_year': genai.protos.Schema(type=genai.protos.Type.STRING)
            },
            required=['id', 'name', 'plot', 'genres', 'cast',
                      'directors', 'production', 'release_year']
        )
    )
    # model = genai.GenerativeModel("gemini-1.5-flash")
    model = genai.GenerativeModel(
        model_name='models/gemini-1.5-flash',
        tools=[movie_object])
    response = model.generate_content(f"""
        Please add id, name, plot, genre, cast, director, production, release_date from this formatted source string to the object,
        Here are few things to take into account, don't summarize the plot and just filter out original plot text, please remove special characters, coments, newline characters and brackets:
        {prompt}
        """)

    return response


def more_movie_response(prompt: str):
    genai.configure(api_key=API_KEY)

    movie_object = genai.protos.FunctionDeclaration(
        name='get_json_from_data_set',
        description=textwrap.dedent(
            """\ generate json data from the source query"""),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'movie1': genai.protos.Schema(type=genai.protos.Type.STRING, description="The name of the first movie related/recommended"),
                'movie2': genai.protos.Schema(type=genai.protos.Type.STRING, description="The name of the first movie related/recommended"),
            },
            required=['movie1', 'movie2']
        )
    )
    # model = genai.GenerativeModel("gemini-1.5-flash")
    model = genai.GenerativeModel(
        model_name='models/gemini-1.5-flash',
        tools=[movie_object])
    response = model.generate_content(f"""
        get 2 different movie suggestions for this plot/related movies based on given moie name, or genre data/cast and crew/backdrop/year, find from your movie knowledge. Given source/prompt string is :
{prompt}
        """)

    return response


def refine_user_prompt(prompt: str):
    genai.configure(api_key=API_KEY)
    # print('gen ai')
    model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
    response = model.generate_content(f"""
        refine this prompt for vector search, don't change the meaning, correct grammar
        {prompt}
        """)
    # print(response.text)
    return response.text

# embeddings = model.encode(texts)

# collection.add(
#     embeddings=embeddings,
#     metadatas=[{"text": text} for text in texts],
#     ids=[str(i) for i in range(len(texts))]
# )


# def get_movie_recommendations(query):
while True:
    print('Enter your interests: ')
    query = input()

    # print(query)
    query = [query]

    refined_text = refine_user_prompt(query)

    # print(refined_text)

    more_results = more_movie_response(refined_text)

    try:
        fc = more_results.candidates[0].content.parts[0].function_call
        fc = json.loads(json.dumps(type(fc).to_dict(fc), indent=4))
        # print(fc)
    except Exception as er:
        print(er)

    # Generate an embedding for the query text
    query_embedding = model.encode(refined_text)

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top 3 similar entries
    )

    fc['args']['movie1'] = fc['args']['movie1'] or None
    fc['args']['movie2'] = fc['args']['movie1'] or None

    if fc['args']['movie1']:
        results['metadatas'][0].insert(
            1, {'name': fc['args']['movie1']})
    if fc['args']['movie2']:
        results['metadatas'][0].insert(
            3, {'name': fc['args']['movie2']})
    # print(results['metadatas'])

    # print(results)

    print('Your top 4 movie suggestions are: ')

    output = {"data": []}

    for i, result in enumerate(results['metadatas'][0]):
        if i % 2:
            if i == 1:
                movie_name = result['name'] or ''
                # output['data'].append({
                #     'name': movie_name,
                #     'year': ''
                # })
                print(f"Result {i + 1}: {movie_name}")
        else:
            data = json.loads(result['text'])
            # output['data'].append({
            #     'name': data['name'],
            #     'year': data['release_year'] or ''
            # })
            if i == 4:
                i = 3
            print(f"Result {i + 1}: {data['name']} ({data['release_year']})")
    # print(output)


# get_movie_recommendations("The Godfather")

# @app.route("/")
# def home():
#     return "Hi from Movie Recommender"


# @app.route('/api/movies', methods=['GET'])
# def movies():
#     query = request.args.get('query')  # Get the 'query' parameter from the URL
#     if not query:
#         # Return error if query is missing
#         return jsonify({"error": "No genre or query specified"}), 400

#     print(query)

#     # Call the function with the query
#     results = get_movie_recommendations(query)
#     return jsonify(results)  # Return the result as JSON


# if __name__ == '__main__':
#     app.run(debug=True)
