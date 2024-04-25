
filepath = joinpath(homedir(), "Desktop", "ECON832")

using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
## add Plots, Random, Combinatorics, LinearAlgebra, JuMP, Ipopt, Flux, Statistics, Parameters
using Plots
using CategoricalArrays
using DataFrames


addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
  #using Gurobi
  #using KNITRO
  #using Ipopt
end

##Machine Learning
using Flux

# Read in the second CSV file
traindata = CSV.read(joinpath(filepath, "All estimation raw data.csv"), DataFrame)

# Read in the first CSV file
testdata = CSV.read(joinpath(filepath, "raw-comp-set-data-Track-2.csv"), DataFrame)

# Drop RT Column from both dataframes
select!(traindata, Not(:RT))
select!(testdata, Not(:RT))
select!(traindata, Not(:SubjID))
select!(testdata, Not(:SubjID))

# Convert non-numeric columns to categorical variables and then to integer codes
non_numeric_columns = [:Location, :Gender, :Condition, :LotShapeA, :LotShapeB, :Amb, :Button, :Feedback]

for col in non_numeric_columns
    traindata[!, col] = categorical(traindata[!, col])
    traindata[!, col] = levelcode.(traindata[!, col])
end

for col in non_numeric_columns
    testdata[!, col] = categorical(testdata[!, col])
    testdata[!, col] = levelcode.(testdata[!, col])
end

@everywhere model="RUM"   # put "LA", "RCG", or "RUM"

## Common parameters
model=="RUM" ? dYu=2 : dYu=2        
dYm=2                               # Number of varying options in menus
Menus=collect(powerset(vec(1:dYm))) # Menus

using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

# Concatenate traindata and testdata
X1 = vcat(traindata, testdata)

# Create arrays of indices
train_indices = 1:510750
test_indices = 510751:size(X1, 1)

function get_processed_data(args)
    labels = string.(X1.B)
    features = Matrix(select(X1, Not(:B)))'

    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)


    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]
    
    X_test = normed_features[:, test_indices]
    y_test = onehot_labels[:, test_indices]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:2)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 37 features as inputs and outputting 6 probabiltiies,
    # one for each lottery.
    ##Create a traditional Dense layer with parameters W and b.
    ##y = σ.(W * x .+ b), x is of length 37 and y is of length 6.
    model = Chain(
    Dense(27, 10, relu),
    Dense(10, 2),
    softmax
)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end


function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.1

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
    ##Loss function
    println("Loss test data")
    loss(x, y) = logitcrossentropy(model(x), y)
    display(loss(X_test,y_test))
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)

##########################################################################################################################
# Model With Risk and Attention 

# Read in the second CSV file
traindata = CSV.read(joinpath(filepath, "All estimation raw data.csv"), DataFrame)

# Read in the first CSV file
testdata = CSV.read(joinpath(filepath, "raw-comp-set-data-Track-2.csv"), DataFrame)

############### Risk Aversion Features ################
# 1. Risk Aversion/Acceptance Indicator
traindata[!, :LowVarPref] = traindata[!, :LotNumA] .< traindata[!, :LotNumB]  # Binary feature indicating preference for lower variance
traindata[!, :HighVarPref] = traindata[!, :LotNumA] .> traindata[!, :LotNumB]  # Binary feature indicating preference for higher variance
testdata[!, :LowVarPref] = testdata[!, :LotNumA] .< testdata[!, :LotNumB]  # Binary feature indicating preference for lower variance
testdata[!, :HighVarPref] = testdata[!, :LotNumA] .> testdata[!, :LotNumB]  # Binary feature indicating preference for higher variance

# 2. Skewness Preference 
traindata[!, :PosSkewPref] = (traindata[!, :LotShapeB] .== "L-skew") .& (traindata[!, :LotShapeA] .!= "L-skew")  # Binary feature indicating preference for more positively skewed option
traindata[!, :NegSkewPref] = (traindata[!, :LotShapeB] .!= "L-skew") .& (traindata[!, :LotShapeA] .== "L-skew")  # Binary feature indicating preference for more negatively skewed option
testdata[!, :PosSkewPref] = (testdata[!, :LotShapeB] .== "L-skew") .& (testdata[!, :LotShapeA] .!= "L-skew")  # Binary feature indicating preference for more positively skewed option
testdata[!, :NegSkewPref] = (testdata[!, :LotShapeB] .!= "L-skew") .& (testdata[!, :LotShapeA] .== "L-skew")  # Binary feature indicating preference for more negatively skewed option

############### Attention Features ################

# 1. Order-related Features
traindata[!, :OrderPref] = (traindata[!, :Order] .< 15) .& traindata[!, :B]  # Binary feature indicating preference for Option B in the first half of the sequence
traindata[!, :FirstChoiceB] = (traindata[!, :Trial] .== 1) .& traindata[!, :B]  # Binary feature indicating whether Option B was chosen in the first trial of each game
testdata[!, :OrderPref] = (testdata[!, :Order] .< 15) .& testdata[!, :B]  # Binary feature indicating preference for Option B in the first half of the sequence
testdata[!, :FirstChoiceB] = (testdata[!, :Trial] .== 1) .& testdata[!, :B]  # Binary feature indicating whether Option B was chosen in the first trial of each game

# 2. Timing-related Features
trial_max = maximum(traindata[!, :Trial])
traindata[!, :TrialNorm] = traindata[!, :Trial] ./ trial_max  # Normalize trial number within each game
testdata[!, :TrialNorm] = testdata[!, :Trial] ./ trial_max  # Normalize trial number within each game

#############################################################################################
# Drop RT Column from both dataframes
select!(traindata, Not(:RT))
select!(testdata, Not(:RT))
select!(traindata, Not(:SubjID))
select!(testdata, Not(:SubjID))

# Convert non-numeric columns to categorical variables and then to integer codes
non_numeric_columns = [:Location, :Gender, :Condition, :LotShapeA, :LotShapeB, :Amb, :Button, :Feedback]

for col in non_numeric_columns
    traindata[!, col] = categorical(traindata[!, col])
    traindata[!, col] = levelcode.(traindata[!, col])
end

for col in non_numeric_columns
    testdata[!, col] = categorical(testdata[!, col])
    testdata[!, col] = levelcode.(testdata[!, col])
end

@everywhere model="RUM"   # put "LA", "RCG", or "RUM"

## Common parameters
model=="RUM" ? dYu=2 : dYu=2        
dYm=2                               # Number of varying options in menus
Menus=collect(powerset(vec(1:dYm))) # Menus

using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

# Concatenate traindata and testdata
X1 = vcat(traindata, testdata)

# Create arrays of indices
train_indices = 1:510750
test_indices = 510751:size(X1, 1)

function get_processed_data(args)
    labels = string.(X1.B)
    features = Matrix(select(X1, Not(:B)))'

    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)


    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]
    
    X_test = normed_features[:, test_indices]
    y_test = onehot_labels[:, test_indices]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:2)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 37 features as inputs and outputting 6 probabiltiies,
    # one for each lottery.
    ##Create a traditional Dense layer with parameters W and b.
    ##y = σ.(W * x .+ b), x is of length 37 and y is of length 6.
    model = Chain(
    Dense(34, 10, relu),
    Dense(10, 2),
    softmax
)

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end


function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.1

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
    ##Loss function
    println("Loss test data")
    loss(x, y) = logitcrossentropy(model(x), y)
    display(loss(X_test,y_test))
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)


