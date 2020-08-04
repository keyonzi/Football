
class Player:
    def __init__(self, name='None', team='None', projection=0, position='None', roster='None', salary=0, ppg=0.0,
                 mult=1.0):
        self.name = name
        self.tm = team
        self.proj = projection
        self.pos = position
        self.ros = roster
        self.sal = salary
        self.ppg = ppg
        self.mult = mult
        self.used = False

    def __str__(self):
        return 'Players Stats(' + self.name + ', ' + self.tm + ', projection= ' + str(self.proj) \
               + ', position= ' + self.pos + ', roster= ' + self.ros + ', salary= ' + str(self.sal) + ', avg ppg= ' \
               + str(self.ppg) + ', multiplier= ' + str(self.mult) + ')'



    def flip_used(self):
        if self.used == True:
            self.used = False
        else:
            self.used = True