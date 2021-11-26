#define Tu_1 13
#define Tu_2 13
#define Tu_3 13
#define Tu_4 13
int data;
void setup()
{
  pinMode(Tu_1, OUTPUT);
  pinMode(Tu_2, OUTPUT);
  pinMode(Tu_3, OUTPUT);
  pinMode(Tu_4, OUTPUT);
  Serial.begin(9600);
  digitalWrite(Tu_1, LOW);
  digitalWrite(Tu_2, LOW);
  digitalWrite(Tu_3, LOW);
  digitalWrite(Tu_4, LOW);
}

void loop()
{
  while( Serial.available() )
  {
    data = Serial.read();

    if (data == '1')
    {
      digitalWrite(Tu_1, HIGH);
    }
    if (data == '0')
    {
      digitalWrite(Tu_1, LOW);
    }

    if (data == '3')
    {
      digitalWrite(Tu_2, HIGH);
    }
    if (data == '2')
    {
      digitalWrite(Tu_2, LOW);
    }

    if (data == '5')
    {
      digitalWrite(Tu_3, HIGH);
    }
    if (data == '4')
    {
      digitalWrite(Tu_3, LOW);
    }

    if (data == '7')
    {
      digitalWrite(Tu_4, HIGH);
    }
    if (data == '6')
    {
      digitalWrite(Tu_4, LOW);
    }
  }
}
