#pragma once

#define VDB_INITIAL_ARRAY_CAPACITY 5
#define VDB_ARRAY_RESIZE_MULTIPLIER 2
#define VDB_INVALID_ID 0
#define VDB_INVALID_CS -10
#define VDB_INVALID_ED -10

typedef double vdb_embedding_t;
typedef unsigned int vdb_vec_id;

typedef struct {
  vdb_embedding_t *vec_embeddings; /* actual vec embeddings */
  unsigned int size;               /* to keep track of vec */
  vdb_vec_id id;                   /* unique identifier */
} vdb_vector;

typedef struct {
  vdb_vector **vector_array; /* array of vectors  */
  unsigned int size;         /* keep track of how many vectors in array */
  unsigned int capacity;     /* keep track of capacity of dynamic array */
  unsigned int vector_size;  /* vector size of all vectors in this instance */
  vdb_vec_id id_counter;     /* id counter to use for added vectors */
  char err[256];             /* space for an informative error message  */
} vdb_instance;

typedef struct {
  float similarity;
  vdb_vec_id vec_id;
} vdb_comparison_result;

typedef struct {
  vdb_comparison_result *vectors; /* array of comparison results */
  unsigned int size;
} vdb_top_n_results;

typedef float (*vdb_comparison_func)(const vdb_vector *vec1,
                                     const vdb_vector *vec2);

/* creates a new vec database instance */
vdb_instance *vdb_new(unsigned int vec_size);
/* handles creating a new vector (allocation) */
vdb_vector *vdb_new_vec(vdb_embedding_t vec_embeddings[], unsigned int size);
/* inserts a vector into the vec database */
vdb_vec_id vdb_insert(vdb_instance *vdb, vdb_vector *vec);
/* returns a pointer intended not to be modified or freed (read-only) */
const vdb_vector *vdb_get(vdb_instance *vdb, vdb_vec_id id);
/* removes from vec database and returns pointer (by extension, ownership) */
vdb_vector *vdb_remove(vdb_instance *vdb, vdb_vec_id id);
/* gets top n vecs using specified comparison function */
vdb_top_n_results vdb_top_n(vdb_instance *vdb,
                            vdb_comparison_func comparison_func,
                            vdb_vector *comparison_vec, unsigned int n);
/* measures cosine similarity between two vectors */
float vdb_measure_cs(const vdb_vector *vec1, const vdb_vector *vec2);
/* measures euclidean distance between two vectors */
float vdb_measure_ed(const vdb_vector *vec1, const vdb_vector *vec2);

/* get error from vdb instance */
char *vdb_get_error(vdb_instance *vdb);

/* frees a non-associated vector */
void vdb_free_vec(vdb_vector *vec);

/* frees the vector database */
void vdb_free(vdb_instance *vdb);
