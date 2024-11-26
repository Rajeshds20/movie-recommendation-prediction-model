import requests
import csv
from urllib.parse import quote

def get_wiki_pages():
    url = 'https://en.wikipedia.org/w/rest.php/v1/page/'

    id = 1

    errs = 0
    
    with open("./imdb_top_1000.csv", mode='r', newline='', encoding='utf-8') as file, open("./output.csv", mode='w', newline='', encoding='utf-8') as output_file:
        csv_reader = csv.DictReader(file)
        fieldnames = ["Id", "Movie_Name", "Plot", "Genres", "Cast", "Director", "Music", "Released_Year"]
        csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in csv_reader:
            # print(row)
            # print(row['Series_Title'])
            # print(id, row['Genre'])
            try:
                movie_name = row['Series_Title']
                response = requests.get(quote(url + movie_name, safe=":/?&="))
                json = response.json()
                # print(json)
                source = json['source']
                if source.startswith('#REDIRECT') or source.startswith('#redirect'):
                    print(movie_name)
                    errs += 1
                    continue
                plot_i = source.index('== Plot ==') if '== Plot ==' in source else source.index('==Plot==')
                plot_j = source.index('== Cast ==') if '== Cast ==' in source else source.index('==Cast==')
                # print(source)
                # plot = source[source.index('== Plot =='): source.index('==Cast==')]
                plot = source[plot_i : plot_j]
                genres = row['Genre']
                cast = source[source.index('starring') : source.index('music')]
                director = source[source.index('director') : source.index('producer')]
                music = source[source.index('music') : source.index('cinematography')]
                released_year = row['Released_Year']
                # print(id, plot[:5], genres[0], cast[0], director[0], music[0], released_year)

                csv_writer.writerow({
                    "Id": id,
                    "Movie_Name": movie_name,
                    "Plot": plot.strip(),
                    "Genres": genres.strip(),
                    "Cast": cast.strip(),
                    "Director": director.strip(),
                    "Music": music.strip(),
                    "Released_Year": released_year.strip()
                })

            except Exception as e:
                print(id, e)
                errs += 1
            finally:
                id += 1
        print(errs)
            
get_wiki_pages()
