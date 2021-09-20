import tensorflow as tf
def max_min_pooling(embedding):
	max_embedding = tf.reduce_max(embedding, axis=-2, keepdims=True)
	min_embedding = -tf.reduce_max(-embedding, axis=-2, keepdims=True)
	embedding = (max_embedding+min_embedding)
	return embedding

def compute_max_sim(query_emb, doc0_emb, doc1_emb):
	pscore = tf.matmul(query_emb, doc1_emb, transpose_b=True) #batch_size, query_len, doc_len
	pscore = tf.where(tf.equal(pscore, 0), tf.zeros_like(pscore)-15, pscore)
	pscore = tf.reduce_max(pscore, axis=-1)
	pscore = tf.reduce_sum(pscore, axis=-1, keepdims=True) #batch_size,1

	nscore = tf.matmul(query_emb, doc0_emb, transpose_b=True) #batch_size, seq_len, seq_len
	nscore = tf.where(tf.equal(nscore, 0), tf.zeros_like(nscore)-15, nscore)
	nscore = tf.reduce_max(nscore, axis=-1)
	nscore = tf.reduce_sum(nscore, axis=-1, keepdims=True) #batch_size,1

	logits = tf.concat([pscore,nscore], axis=1) #batch_size,2
	margins = tf.reduce_sum(pscore) - tf.reduce_sum(nscore)
	return logits, margins



def batch_softmax_loss(query_emb, doc0_emb, doc1_emb):
	batch_size = query_emb.shape[0].value
	doc_emb = tf.concat([doc1_emb, doc0_emb], axis=0) #2*batch_size, output_dim
	logits = tf.matmul(query_emb, doc_emb, transpose_b=True) #batch_size, 2*batch_size
	labels = tf.eye(batch_size, num_columns=2*batch_size, dtype=tf.dtypes.int32)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	return loss
def doc_max_sim_label(query_emb, doc_emb):
	scores = tf.matmul(query_emb, doc_emb, transpose_b=True) #batch_size, query_len, doc_len
	scores = tf.where(tf.equal(scores, 0), tf.zeros_like(scores)-15, scores)
	important_word_label = tf.where(
		tf.equal(tf.reduce_max(scores, axis=-1, keep_dims=True), scores),
		tf.constant(1, shape=scores.shape),
		tf.constant(0, shape=scores.shape)
	)
	important_word_label = tf.reduce_sum(important_word_label, axis=1) #batch_size, doc_len
	important_word_label = tf.where(tf.greater_equal(important_word_label, 1), tf.constant(1, shape=important_word_label.shape), tf.constant(0, shape=important_word_label.shape))
	return important_word_label


def batch_max_sim_softmax_loss(query_emb, doc0_emb, doc1_emb, soft):


	batch_size = doc0_emb.shape[0].value
	doc_length = doc0_emb.shape[1].value
	hidden_size = doc0_emb.shape[2].value

	doc0_emb = tf.reshape(doc0_emb, (-1, hidden_size)) #batch_size*doc_len, hiddien_size
	doc1_emb = tf.reshape(doc1_emb, (-1, hidden_size)) #batch_size*doc_len, hiddien_size
	doc_emb = tf.concat([doc1_emb, doc0_emb], axis=0) #2*batch_size*doc_len, hiddien_size
	doc_emb = tf.tile(doc_emb, (batch_size,1)) #batch_size*2*batch_size*doc_len, hiddien_size
	doc_emb = tf.reshape(doc_emb, (batch_size, -1, hidden_size)) # batch_size, 2*batch_size*doc_len, hiddien_size

	logits = tf.matmul(query_emb, doc_emb, transpose_b=True) #batch_size, query_len, 2*batch_size*doc_len
	logits = tf.reshape(logits, (batch_size, -1, 2*batch_size, doc_length)) #batch_size, query_len, 2*batch_size, doc_len

	#remove_zero:
	logits = tf.where(tf.equal(logits, 0), tf.zeros_like(logits)-15, logits)
	logits = tf.transpose(logits, perm=[0, 2, 1, 3]) #batch_size, 2*batch_size, query_len, doc_len
	logits = tf.reduce_max(logits, axis=-1) #batch_size, 2*batch_size, query_len
	logits = tf.reduce_sum(logits, axis=-1) #batch_size, 2*batch_size



	labels = tf.eye(batch_size, num_columns=2*batch_size, dtype=tf.dtypes.float32)
	anchors = logits*labels
	anchors = tf.reduce_sum(anchors, axis=-1, keepdims=True) #batch_size
	anchors = tf.tile(anchors, (1, 2*batch_size)) #batch_size, 2*batch_size
	margins = logits - anchors
	labels = tf.eye(batch_size, num_columns=2*batch_size, dtype=tf.dtypes.int32)
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits*soft) #soft logits
	return loss, logits, margins

def compute_max_sim_score(query_embs, doc_embs):
	total_scores = 0
	for i in range(len(doc_embs)):
		doc_emb = doc_embs[i]
		query_emb = query_embs[i]
		batch_size = doc_emb.shape[0].value
		doc_length = doc_emb.shape[1].value
		hidden_size = doc_emb.shape[2].value

		score = tf.matmul(query_emb, doc_emb, transpose_b=True) #batch_size, query_len, doc_length
		score = tf.reduce_max(score, axis=-1)


		score = tf.reduce_sum(score, axis=-1) #batch_size
		total_scores+=score
	return total_scores

def compute_pooling_score(query_emb, doc_emb):
	score = query_emb*doc_emb
	score = tf.reduce_sum(score, axis=-1) #batch_size
	return score