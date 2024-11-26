import csv
import google.generativeai as genai
import textwrap
import os
import json
import logging
logging.basicConfig(level=logging.DEBUG)


def generate_prompt(id, movie_name, plot, genre, cast, directors, music, release_year):
    prompt = f"Id: {id}\nMovie Name: {movie_name}\nPlot: {plot}\nGenre: {genre}\nCast: {cast}\nDirectors: {directors}\nMusic: {music}\nRelease Year: {release_year}"
    return prompt


def call_gemini(prompt: str):
    genai.configure(api_key="AIzaSyAdMBWB7s2omSLD6kqKr3rWmRb7eMPuWwE")
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


# with open('opt.txt', 'r',  encoding="utf-8") as filename:
#     movie_data = filename.read()
#     result = call_gemini(movie_data)
#     fc = result.candidates[0].content.parts[0].function_call
#     print(json.dumps(type(fc).to_dict(fc), indent=4))


with open('output.csv', 'r', encoding="utf-8") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        result = call_gemini(generate_prompt(row['Id'], row['Movie_Name'], row['Plot'],
                             row['Genres'], row['Cast'], row['Director'], row['Music'], row['Released_Year']))
        try:
            fc = result.candidates[0].content.parts[0].function_call
            fc = json.loads(json.dumps(type(fc).to_dict(fc), indent=4))
            # result_json_movie.append(fc)
            # try:
            #     with open('results.json', 'r') as json_file:
            #         data = json.load(json_file)
            #         if not isinstance(data, list):
            #             raise ValueError("JSON root element must be a list.")
            # except FileNotFoundError:
            #     data = []
            # except json.JSONDecodeError:
            #     print("Error: Invalid JSON format in file. Initializing an empty list.")
            #     data = []
            # data.append(fc)
            # with open('results.json', 'w') as json_file:
            #     json_file.dump(data, json_file, indent=4)
            try:
                with open('results.json', 'r') as json_file:
                    data = json.load(json_file)
                    if not isinstance(data, list):
                        raise ValueError("JSON root element must be an array.")
                    # print(data)
                data.append(fc)
                with open('results.json', 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                print(
                    f"Appended result for row ID {row['Id']} to results.json")
            except Exception as e:
                print(e)
        except Exception as er:
            print(er)
