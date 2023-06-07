tickets = trainData[['Ticket']]
#numTickets = trainData.loc[trainData['Ticket'].str.split(r'\d+')]
#tickets = tickets.str.split(r'\d+', expand=True)
tickets
s = tickets['Ticket'].str.split(r'(?=(?:\b\d+))', n=1, expand=True)
s[0] = s[0].str.replace('.', '').str.lower().str.strip()
s = s[0].str.split(r'/', n=1, expand=True).join(s[1], rsuffix='r')
display(s)
display(s.iloc[:, 0].unique())
display(s.iloc[:, 1].unique())
s.columns = ['TPrefix', 'TNumber']
display(s)
tickets
