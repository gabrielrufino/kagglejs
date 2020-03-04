const brain = require('brain.js')
const fs = require('fs')
const parse = require('csv-parse/lib/sync')
const stringify = require('csv-stringify/lib/sync')

/**
 * Preparing the Dataset
 */

const trainDataFile = fs.readFileSync('./data/train.csv', 'utf8')
const trainData = parse(trainDataFile, {
  columns: true,
  skip_empty_lines: true
})

/**
 * Mapping non numerics data to numeric data
 */

const sexCode = {
  female: Math.random(),
  male: Math.random()
}

const embarkedCode = {
  C: Math.random(),
  Q: Math.random(),
  S: Math.random()
}

/**
 * Starting the initial neural network
 */

const network = new brain.NeuralNetwork({
  hiddenLayers: [8, 8]
})

/**
 * Traning the network
 */

const networkData = trainData.map(passenger => ({
  input: {
    Pclass: passenger.Pclass,
    Sex: sexCode[passenger.Sex],
    Age: Number(passenger.Age),
    SibSp: passenger.SibSp,
    Parch: passenger.Parch,
    Fare: passenger.Fare,
    Embarked: embarkedCode[passenger.Embarked]
  },
  output: { Survived: Number(passenger.Survived) }
}))

network.train(networkData, {
  log: true,
  learningRate: 0.001
})

/**
 * Preparing the test dataset
 */

const testDataFile = fs.readFileSync('./data/test.csv', 'utf8')
const testData = parse(testDataFile, {
  columns: true,
  skip_empty_lines: true
})

/**
 * Predictions on the test  dataset
 */

const predictions = testData.map(passenger => {
  const result = network.run({
    Pclass: passenger.Pclass,
    Sex: sexCode[passenger.Sex],
    Age: Number(passenger.Age),
    SibSp: passenger.SibSp,
    Parch: passenger.Parch,
    Fare: passenger.Fare,
    Embarked: embarkedCode[passenger.Embarked]
  })

  return {
    PassengerId: passenger.PassengerId,
    Survived: Math.round(result.Survived)
  }
})

/**
 * Generating the output file
 */

const output = stringify(predictions, {
  header: true,
  columns: ['PassengerId', 'Survived']
})

fs.writeFileSync('./output.csv', output)
