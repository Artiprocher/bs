#ifndef ML_TIMER
#define ML_TIMER

namespace Timer{
int timer=0;
void reset(){
    timer=0;
}
char* getTime(){
    time_t now=time(0);
    char *t=ctime(&now);
    for(char *c=t;*c!='\0';c++){
        if(*c==':')*c='-';
        else if(*c==' ')*c='_';
    }
    return t;
}
}

#endif