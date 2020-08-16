import json
import time
from urllib.request import urlopen
from urllib.error import URLError

import numpy
import requests
from bs4 import BeautifulSoup

from FFNN import FFNN


def get_matches_page(link):
    request = requests.get(link, headers={'User-Agent': '***'}).content
    matches_page = BeautifulSoup(str(request), 'html.parser')
    return matches_page


def get_team_players_links(team_page):
    team_players_links = team_page.find('div', {'class': '***'})
    team_players_links = team_players_links.findAll('a', {'class': '***'})
    return team_players_links


def get_team_links(team_page, site_link):
    team_players_links = get_team_players_links(team_page)

    team_links = []
    for team_link in team_players_links:
        team_links.append(site_link + team_link.get('***'))
    return team_links


def get_team_link(matches_page, site_link):
    team_link = matches_page.find('div', {'class': '***'})
    team_link = team_link.find('a').get('***')
    team_link = site_link + team_link
    return team_link


def get_team_rates(team_page, map_names):
    team_rates = []
    all_maps_container = team_page.findAll('div', {'class': '***'})

    for iterator in range(len(map_names)):
        for map_container in all_maps_container:
            processing_map = map_container.find('div', {'class': '***'})
            map_rate = map_container.find('div', {'class': '***'})

            if map_names[iterator] == processing_map.text:
                team_rates.append(map_rate.text)
    return team_rates


def get_team_world_rank(team_page):
    team_world_rank = team_page.findAll('div', {'class': '***'})[0]
    team_world_rank = team_world_rank.findAll('a')
    if len(team_world_rank) < 1:
        return False
    team_world_rank = team_world_rank[0].text
    team_world_rank = int(team_world_rank[1:])
    return team_world_rank


def get_team_in_top(team_page):
    team_in_top = team_page.findAll('div', {'class': '***'})[1]
    if len(team_in_top) < 1:
        return False
    team_in_top = team_in_top.find('span').text
    team_in_top = int(team_in_top)
    return team_in_top


def get_team_average_age(team_page):
    team_average_age = team_page.findAll('div', {'class': '***'})[2]
    if len(team_average_age) < 1:
        return False
    team_average_age = team_average_age.find('span').text
    team_average_age = float(team_average_age)
    return team_average_age


def get_team_players_stats(team_page, site_link):
    team_links = get_team_links(team_page, site_link)

    team_players_stats = []
    try:
        for team_link in team_links:
            team_player_page = get_matches_page(team_link)

            cols = team_player_page.findAll('div', {'class': '***'})
            first_value = float(cols[1].find('span', {'class': '***'}).text)
            second_value = float(cols[4].find('span', {'class': '***'}).text)

            team_players_stats.append([first_value, second_value])
    except Exception:
        return False
    return team_players_stats


def check_if_maps_are_chosen(matches_page, active_duty_map_pool):
    maps = matches_page.find('div', {'class': '***'}).findAll('div', {'class': '***'})
    maps_are_chosen = True

    for Map in maps:
        if Map.text not in active_duty_map_pool:
            maps_are_chosen = False
    return maps_are_chosen


def get_teams_names(matches_page):
    teams = matches_page.findAll('div', {'class': '***'})
    team1, team2 = teams[0].text, teams[1].text
    return team1, team2


def get_maps_names(matches_page):
    maps = matches_page.findAll('div', {'class': '***'})

    map_names = []
    for Map in maps:
        map_name = Map.find('div', {'class': '***'}).text
        map_names.append(map_name)

    if '***' in maps or len(map_names) < 1:
        return False
    else:
        return map_names


def collect_team_parameters(matches_page, map_names, site_link, team):
    team_link = get_team_link(matches_page, site_link)
    team_page = get_matches_page(team_link + '***')
    team_rates = get_team_rates(team_page, map_names)

    team_world_rank = get_team_world_rank(team_page)
    team_in_top = get_team_in_top(team_page)
    team_average_age = get_team_average_age(team_page)
    team_players_stats = get_team_players_stats(team_page, site_link)

    if False in (team_world_rank, team_in_top, team_average_age, team_players_stats) or \
            len(team_rates) != len(map_names) or \
            len(team_page.findAll('div', {'class': '***'})) < 3:
        return False

    team_parameters = [team, team_world_rank, team_in_top, team_average_age,
                       team_players_stats[0][0], team_players_stats[0][1]]

    # A pause preventing Error 1015
    time.sleep(1)
    return team_parameters, team_rates


def gather_map_info(team1_parameters, team2_parameters, map_names, team1_rates, team2_rates, game):
    map_info = []
    for p in team1_parameters:
        map_info.append(p)
    for p in team2_parameters:
        map_info.append(p)
    map_info.append(map_names[game])
    map_info.append(team1_rates[game])
    map_info.append(team2_rates[game])
    return map_info


def normalize_parameters(map_info):
    row = map_info
    data_set = row[1:14] + row[15:17]
    data_set = list(data_set)

    data_set[0] = data_set[0] * 0.01
    data_set[0] = float(str(data_set[0])[:5])
    data_set[1] = data_set[1] * 0.001
    data_set[1] = float(str(data_set[1])[:5])
    data_set[2] = data_set[2] * 0.01
    data_set[2] = float(str(data_set[2])[:5])
    data_set[13] = data_set[13] * 0.01
    data_set[13] = float(str(data_set[13])[:5])
    data_set[14] = data_set[14] * 0.001
    data_set[14] = float(str(data_set[14])[:5])
    data_set[15] = data_set[15] * 0.01
    data_set[15] = float(str(data_set[15])[:5])

    inputs = numpy.array(data_set, ndmin=2).T
    inputs = [x[0] for x in inputs]
    return inputs


def get_map_pool():
    map_pool = []

    page = get_matches_page('https://***')

    for m in page.find('table', {'class': '***'}). \
            findAll('table', {'class': '***'}):
        if '***' in m.text:
            for tr in m.find('***').findAll('***'):
                if '***' in tr.text:
                    for a in tr.find('td', {'class': '***'}).findAll('a'):
                        if a.text == '***':
                            map_pool.append('***')
                        else:
                            map_pool.append(a.text)
    return map_pool


def read_weights(weights, path):
    with open(path) as f:
        data = json.load(f)
        for key in data.keys():
            layer_weights = []
            for item in data[key]:
                mass = []
                for cell in item.values():
                    mass.append(cell)
                layer_weights.append(mass)
            weights.append(numpy.asfarray(layer_weights))
    return weights


def recreate_nodes_from_weights(weights):
    layer_nodes = []
    for number, w in enumerate(weights):
        if number == 0:
            layer_nodes.append(len(w[0]))
        layer_nodes.append(len(w))
    return layer_nodes


def scrape_live_matches_links(matches_page):
    links = []
    if len(matches_page.findAll('div', {'class': '***'})) >= 1:
        live_matches_section = matches_page.find('div', {'class': '***'}). \
            findAll('div', {'class': '***'})
        for match in live_matches_section:
            link = match.find('a', {'class': '***'}).get('***')
            links.append(link)
    return links


def collect_predicts(links, site_link, map_pool, nn, matches_found):
    predicts = []

    if links is not []:
        for link in links:
            matches_page = get_matches_page(site_link + link)

            # Check if playing maps are already chosen
            if check_if_maps_are_chosen(matches_page, map_pool) is False:
                continue

            # Getting of team names
            team1, team2 = get_teams_names(matches_page)

            # Getting of maps' names
            map_names = get_maps_names(matches_page)

            # Getting of teams' parameters
            parameters = collect_team_parameters(matches_page, map_names, site_link, team1)
            parameters2 = collect_team_parameters(matches_page, map_names, site_link, team2)

            if parameters is False or parameters2 is False:
                continue
            else:
                team1_parameters, team1_rates = parameters
                team2_parameters, team2_rates = parameters2

            for game in range(len(map_names)):
                # Gathering info
                map_info = gather_map_info(team1_parameters, team2_parameters,
                                           map_names, team1_rates, team2_rates, game)
                # Parameters normalization
                inputs = normalize_parameters(map_info)
                # Predicting results
                final_outputs = nn.ask_net(inputs)

                predict = f'{team1} : {team2}:{map_names[game]}: Team {numpy.argmax(final_outputs) + 1} will win.'
                predicts.append(predict)
                matches_found = True

            time.sleep(1)
    return predicts, matches_found


def get_predicts_results():
    predicts = []
    matches_found = False
    error_code = 0

    # Checking for an internet connection
    try:
        urlopen("http://google.com")
    except URLError:
        print("Network currently down!")
        error_code = 2  # EC = 2 none internet connection detected
        return predicts, matches_found, error_code

    # Getting map pool
    try:
        map_pool = get_map_pool()
    except Exception:
        error_code = 4  # EC = 4 cannot scrap active duty maps
        return predicts, matches_found, error_code

    # Reading weights from file
    weights = []
    try:
        weights = read_weights(weights, './weights/***.json')
    except FileNotFoundError:
        error_code = 3  # EC = 3 Could not find weights.txt file
        return predicts, matches_found, error_code

    # Initializing an instance of a neural net's class
    nn = FFNN(layer_nodes=recreate_nodes_from_weights(weights))
    nn.weights = weights

    matches_link = 'https://***'
    site_link = 'https://***'

    matches_page = get_matches_page(matches_link)
    links = scrape_live_matches_links(matches_page)
    predicts, matches_found = collect_predicts(links, site_link, map_pool, nn, matches_found)

    return predicts, matches_found, error_code
