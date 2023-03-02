import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class G002HW1
{
    public static void main(String[] args) throws IOException {

        //control on CLI arguments
        if (args.length != 4)
        {
            throw new IllegalArgumentException(
                    "You must input K (n of partitions), " +
                            "H (best-H will be displayed), " +
                            "a string corresponding to a country code in the dataset" +
                            "and the path to the input file");
        }

        // SPARK SETUP
        SparkConf conf = new SparkConf(true).setAppName("G002HW1F");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING: K H S path
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);
        String S = args[2];
        int H = Integer.parseInt(args[1]);
        //TASK 1
        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();
        long numdocs;
        Random randomGenerator = new Random();
        numdocs = rawData.count();
        System.out.println("Number of rows = " + numdocs);

        //TASK 2
        JavaPairRDD<String, Integer> productCustomer;

        productCustomer = rawData
                .flatMapToPair((document) ->{ // <-- Map phase (R1)
                    String[] tokens = document.split(",");
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();

                    if(Integer.parseInt(tokens[3]) >0
                            && (tokens[7].equals(S) || S.equals("all")))
                    {
                        String s = tokens[1] +","+ tokens[6];
                        pairs.add(new Tuple2<String, Integer>(s, 0));
                    }
                    return pairs.iterator();
                })
                .reduceByKey((x, y) -> 0)   // <-- Reduce phase (R1)
                //now productCustomer contains ((productID,CustomerID), 0)
                //that is we have a dummy value.
                .flatMapToPair((couple) -> {
                    String[] tokens = couple._1().split(",");
                    String key = tokens[0];
                    Integer value = Integer.parseInt(tokens[1]);
                    ArrayList<Tuple2<String, Integer>> pair = new ArrayList<>();
                    pair.add(new Tuple2<String,Integer>(key, value));
                    return pair.iterator();
                });
        System.out.println("Product-Customer Pairs = " + productCustomer.count());

        //TASK 3

        JavaPairRDD<String, Integer> productPopularity1 = productCustomer
                .mapPartitionsToPair((pairs) -> {
                    HashMap<String, Integer> counts = new HashMap<>(); //key: product; value: count of (unique) buyers
                    while (pairs.hasNext()) {
                        Tuple2<String, Integer> cur = pairs.next();
                        counts.put(cur._1(), 1 + counts.getOrDefault(cur._1(), 0)); //add new pair or increase count by 1
                    }
                    ArrayList<Tuple2<String, Integer>> to_return = new ArrayList<>();   //will contain the (product, popularity)
                    // pairs for the partition
                    for (Map.Entry<String, Integer> e : counts.entrySet()) {
                        to_return.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return to_return.iterator();
                }) //I get a partial count for each partition
                .reduceByKey((count1, count2) -> count1+count2); //now I aggregate all partial counts


        //TASK 4
        /*
        Implement partition split manually assigning keys "on the fly"
         */
        JavaPairRDD<String, Integer> productPopularity2 = productCustomer
                .mapToPair(pair -> new Tuple2<String, Integer>(pair._1, 1)) //returns JavaPairRDD<String, Integer>
                .groupBy((pair) -> randomGenerator.nextInt(K)) //returns JavaPairRDD<Integer, Iterable<Tuple2<String, Integer>>>
                .flatMapToPair((element) -> { //element is a Tuple2<Integer, Iterable<Tuple2<String, Integer>>>
                    HashMap<String, Integer> counts = new HashMap<>();
                    for (Tuple2<String, Integer> cur : element._2) { //generalized for loop for iterables
                        counts.put(cur._1(), 1 + counts.getOrDefault(cur._1(), 0));
                    }
                    ArrayList<Tuple2<String, Integer>> to_return = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : counts.entrySet()) {
                        to_return.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return to_return.iterator();
                }) //returns a JavaPairRDD<String, Integer>
                .reduceByKey((count1, count2) -> count1+count2);//finally computes the sum of all counts having the same key

        // task 5
        /*
        If H>0:
        Saves in a list and prints the ProductID and Popularity of the H products with highest Popularity.
        */
        if(H > 0) {
            List<Tuple2<Integer, String>> task = productPopularity2.mapPartitionsToPair((pairs) -> {
                HashMap<Integer, String> counts = new HashMap<>();
                while (pairs.hasNext()) {
                    Tuple2<String, Integer> cur = pairs.next();
                    counts.put(cur._2(), cur._1);
                }
                ArrayList<Tuple2<Integer, String>> to_return = new ArrayList<>();
                for (Map.Entry<Integer, String> e : counts.entrySet()) {
                    to_return.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                return to_return.iterator();
            }).sortByKey(false).take(H);

            System.out.println("\nTop 5 Products and their Popularities");
            for(Tuple2<Integer, String> t : task)
            {
                System.out.print("Product " + t._2()+" ");
                System.out.print("Popularity "+ t._1()+"; ");
            }

        }

        // task 6
        /*
        If H==0: Collects all pairs of productPopularity1 into a list
                and print all of them, in increasing lexicographic order of ProductID.
                Repeats the same thing using productPopularity2.
        */
        if(H == 0)
        {
            System.out.println("\nproductPopularity1: ");
            //ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
            for(Tuple2<String, Integer> t : productPopularity1.sortByKey().collect())
            {

                System.out.print("Product: " + t._1() + " ");
                System.out.print("Popularity: " + t._2() + "; ");
            }
            System.out.println("\nproductPopularity2: ");
            for(Tuple2<String, Integer> t : productPopularity2.sortByKey().collect())
            {

                System.out.print("Product: " + t._1() + " ");
                System.out.print("Popularity: " + t._2() + "; ");
            }
        }
    }
}
