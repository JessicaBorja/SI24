
En esta parte del código, al épsilon siempre le estas restando '-.005' pero -(-0.005) en realidad es + entonces le estas sumando siempre a la exploración. La exploración está muy alta y es probable que nunca converja.
```
    # TODO: Implementa algun código para reducir la exploración del agente conforme aprende
    # puedes decidir hacerlo por episodio, por paso del tiempo, retorno promedio, etc.
        a=0
        a-=.005
        agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.9-a)
```